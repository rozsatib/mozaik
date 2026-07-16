import ast
import gc
import json
import os
import shutil

import numpy as np
import psutil
import yaml
from scipy.ndimage import gaussian_filter1d

# Post-blank duration (ms) appended after each image, mirroring the simulation's RandomizedExperanto
# experiment. MUST equal the post-blank in mozaik/experiments/vision.py (InternalStimulus); the spike
# and screen timelines are aligned exactly, so a drift here desyncs them. Single source on this side.
POST_BLANK_MS = 49.0


def load_tier_reference(combined_meta_path):
    """Build a condition_hash → tier mapping from an existing combined_meta.json.

    When multiple entries share the same condition_hash, the first non-blank
    tier encountered wins. This lets you pass the original (mouse) dataset's
    combined_meta.json to preserve its train/validation/test splits in a
    Mozaik re-export.

    Parameters
    ----------
    combined_meta_path : str
        Path to a ``screen/combined_meta.json`` file.

    Returns
    -------
    dict
        Mapping of ``{condition_hash: tier}``.
    """
    with open(combined_meta_path, "r") as f:
        meta = json.load(f)
    ref = {}
    for entry in meta.values():
        ch = entry.get("condition_hash")
        tier = entry.get("tier")
        if ch and tier and ch not in ref:
            ref[ch] = tier
    return ref


def get_process_memory():
    """Returns current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


class MozaikTrialExporter:
    """
    Stateful exporter to handle batch processing of Mozaik DataStoreViews
    into a single large output file.
    Supports appending to an existing export if append_mode=True.

    Parameters
    ----------
    stim_name_key : str
        Stimulus parameter that names each stimulus (used for ``stimuli_order`` and to detect
        blanks: a segment is treated as blank when this parameter is absent). Default
        ``"movie_name"`` reproduces the historical behaviour.
    group_by_key : str
        Stimulus parameter used to select which segments are exported together. Default
        ``"trial"`` reproduces the historical behaviour; e.g. a long movie presented in chunks
        could instead be grouped by its per-chunk start time.
    group_value : optional
        Value of ``group_by_key`` selecting this export; defaults to ``trial_id`` so the default
        keys/values (and the written ``meta.yml``) are byte-identical to the previous
        trial-based export.
    """

    def __init__(
        self,
        output_dir,
        trial_id,
        sampling_rate=1000.0,
        smooth_param=None,
        append_mode=False,
        stim_name_key="movie_name",
        group_by_key="trial",
        group_value=None,
    ):
        self.output_dir = output_dir
        self.trial_id = trial_id
        self.sampling_rate = sampling_rate
        self.smooth_param = smooth_param
        # Which stimulus parameter names each stimulus, and which parameter (and value) selects
        # the segments to export together. Defaults reproduce the historical trial/movie_name
        # behaviour exactly.
        self.stim_name_key = stim_name_key
        self.group_by_key = group_by_key
        self.group_value = trial_id if group_value is None else group_value

        self.meta_segments = []
        self.all_unit_spike_lists = None
        self.num_units = 0
        self.total_bins_accumulated = 0
        self.current_time_offset = 0.0

        os.makedirs(self.output_dir, exist_ok=True)

        # Check for existing data if append_mode is active
        if append_mode:
            self._load_existing_state()
        else:
            print(f"Exporter initialized. Target: {self.output_dir} (New Export)")

    def _load_existing_state(self):
        """Attempts to load state from existing meta.yaml and spikes.npy."""
        meta_path = os.path.join(self.output_dir, "meta.yml")
        spikes_path = os.path.join(self.output_dir, "spikes.npy")

        if os.path.exists(meta_path) and os.path.exists(spikes_path):
            print(f"Loading existing data from {self.output_dir}...")

            # 1. Load Metadata
            with open(meta_path, "r") as f:
                meta_data = yaml.safe_load(f)

            # Verify compatibility
            if meta_data.get("sampling_rate") != self.sampling_rate:
                raise ValueError("Sampling rate mismatch with existing data.")
            if meta_data.get("trial_id") != self.trial_id:
                print(
                    f"Warning: Existing data has trial_id {meta_data.get('trial_id')}, expected {self.trial_id}"
                )

            self.num_units = meta_data["n_signals"]
            self.meta_segments = meta_data.get("stimuli_order", [])
            spike_indices = meta_data["spike_indices"]

            # end_time is in seconds; convert back to ms for internal offset
            self.current_time_offset = meta_data["end_time"] * 1000.0
            bin_size_ms = 1000.0 / self.sampling_rate
            self.total_bins_accumulated = int(
                np.ceil(self.current_time_offset / bin_size_ms)
            )

            # 2. Load Spikes and Reconstruct Lists
            # We must load the full array to append to it efficiently in memory
            # Spikes on disk are in seconds; convert back to ms for internal accumulation
            flat_spikes = np.load(spikes_path) * 1000.0

            self.all_unit_spike_lists = []

            # Reconstruct list of lists using the CSR indices
            # spike_indices = [0, end_0, end_1, ..., end_{N-1}] (length N+1)
            for i in range(self.num_units):
                start = spike_indices[i]
                end = spike_indices[i + 1]
                # Copying to list allows appending later
                unit_spikes = flat_spikes[start:end]
                self.all_unit_spike_lists.append([unit_spikes])

            print(
                f"Resumed from offset: {self.current_time_offset} ms with {self.num_units} units."
            )
            del flat_spikes  # Free raw buffer
            gc.collect()
        else:
            print("No existing data found to append. Starting fresh.")

    def _in_group(self, raw):
        """Whether a segment whose ``group_by_key`` value is ``raw`` belongs to this export.

        Reproduces the historical integer trial comparison for the default ("trial") key and
        falls back to a direct equality test for other (e.g. float start-time) grouping keys.
        A segment missing the grouping parameter (``raw is None``) is not selected.
        """
        if raw is None:
            return False
        try:
            return int(raw) == int(self.group_value)
        except (TypeError, ValueError):
            return raw == self.group_value

    def process_batch(self, dsv_or_list):
        """
        Process a batch of DSVs. Accumulates spike times in memory lists
        and updates metadata.
        """
        import sys
        import time

        if isinstance(dsv_or_list, (list, tuple)):
            dsvs = dsv_or_list
        else:
            dsvs = [dsv_or_list]

        if not dsvs:
            return

        batch_t0 = time.time()
        print(
            f"Processing batch of {len(dsvs)} DSVs... (Mem: {get_process_memory():.2f} MB)",
            flush=True,
        )

        # 1. Scan Metadata for this batch — segments are kept in chronological
        #    order (as stored by Mozaik) so the time offsets stay aligned with
        #    the screen timeline.  Both PixelMovieExperanto and InternalStimulus
        #    (blank) segments are included.
        scan_t0 = time.time()
        batch_segments = []
        for dsv_i, dsv in enumerate(dsvs):
            dsv_t0 = time.time()
            segment_refs = dsv.get_segments()
            n_segs_in_dsv = 0
            for seg in segment_refs:
                try:
                    if isinstance(seg.annotations["stimulus"], str):
                        stim_params = ast.literal_eval(seg.annotations["stimulus"])
                    else:
                        stim_params = seg.annotations["stimulus"]
                except (ValueError, SyntaxError):
                    print(
                        f"Warning: Parse error for segment {seg}. Skipping.", flush=True
                    )
                    continue

                if self._in_group(stim_params.get(self.group_by_key)):
                    stim_name = stim_params.get(self.stim_name_key)
                    batch_segments.append(
                        {
                            "segment": seg,
                            "stim_name": stim_name if stim_name else "blank",
                            "duration": stim_params["duration"],
                            "is_blank": stim_name is None,
                        }
                    )
                    n_segs_in_dsv += 1
            print(
                f"  DSV {dsv_i}: {n_segs_in_dsv} segments scanned in {time.time() - dsv_t0:.1f}s",
                flush=True,
            )

        n_blanks = sum(1 for s in batch_segments if s["is_blank"])
        n_stim = len(batch_segments) - n_blanks
        print(
            f"  Scan total: {time.time() - scan_t0:.1f}s — {len(batch_segments)} segments "
            f"({n_stim} stimulus, {n_blanks} blank)",
            flush=True,
        )

        if not batch_segments:
            print("No matching segments in this batch.", flush=True)
            return

        bin_size_ms = 1000.0 / self.sampling_rate

        # 2. Stream Process this Batch
        t_load = 0.0  # time in get_spiketrains()
        t_loop = 0.0  # time in per-unit spike extraction
        n_processed = 0
        seg_t0 = time.time()

        for meta in batch_segments:
            seg = meta["segment"]
            seg_duration = meta["duration"]
            num_seg_bins = int(np.ceil(seg_duration / bin_size_ms))
            is_blank = meta["is_blank"]

            # Store metadata
            self.meta_segments.append(meta["stim_name"])

            if is_blank:
                # Blank segments (InternalStimulus): advance the timeline
                # but skip the expensive per-unit spike extraction.
                # The dataloader filters blanks via screen metadata anyway.
                self.current_time_offset += seg_duration
                self.total_bins_accumulated += num_seg_bins
                # Don't call seg.release() here — spiketrains were never
                # loaded, so release() would fail with AttributeError.
                continue

            # Load spikes once per segment
            t0 = time.time()
            spiketrains = seg.get_spiketrains()
            t_load += time.time() - t0

            # Initialize unit count on first segment seen
            if self.all_unit_spike_lists is None:
                self.num_units = len(spiketrains)
                self.all_unit_spike_lists = [[] for _ in range(self.num_units)]
                print(f"Initialized for {self.num_units} units.", flush=True)

            limit_units = min(self.num_units, len(spiketrains))
            offset = self.current_time_offset

            # Tight loop — avoid repeated attribute lookups
            t0 = time.time()
            _lists = self.all_unit_spike_lists
            _trains = spiketrains
            _dur = seg_duration
            for unit_idx in range(limit_units):
                spikes = np.asarray(_trains[unit_idx])
                if len(spikes) == 0:
                    continue
                valid = spikes[spikes < _dur]
                if len(valid):
                    valid += offset
                    _lists[unit_idx].append(valid)
            t_loop += time.time() - t0

            # Update global offsets
            self.current_time_offset += seg_duration
            self.total_bins_accumulated += num_seg_bins
            n_processed += 1

            if hasattr(seg, "release"):
                seg.release()

            # Progress every 50 stimulus segments
            if n_processed % 50 == 0:
                elapsed = time.time() - seg_t0
                avg_load = t_load / n_processed
                avg_loop = t_loop / n_processed
                eta = (n_stim - n_processed) * (elapsed / n_processed)
                print(
                    f"  [{n_processed}/{n_stim}] {elapsed:.0f}s elapsed, "
                    f"ETA {eta:.0f}s — "
                    f"load: {t_load:.1f}s (avg {avg_load:.2f}s/seg), "
                    f"loop: {t_loop:.1f}s (avg {avg_loop:.3f}s/seg), "
                    f"offset: {self.current_time_offset/1000:.1f}s, "
                    f"mem: {get_process_memory():.0f} MB",
                    flush=True,
                )

        total_t = time.time() - batch_t0
        print(
            f"Batch complete in {total_t:.1f}s — "
            f"load: {t_load:.1f}s ({t_load/total_t*100:.0f}%), "
            f"loop: {t_loop:.1f}s ({t_loop/total_t*100:.0f}%), "
            f"other: {total_t - t_load - t_loop:.1f}s. "
            f"Offset: {self.current_time_offset:.0f} ms. "
            f"(Mem: {get_process_memory():.2f} MB)",
            flush=True,
        )

    def finalize(self):
        """
        Writes the final spikes.npy and meta.yaml combining all batches.
        """
        print("Finalizing export...")
        print(f"Memory before concat: {get_process_memory():.2f} MB")

        # Pre-compute total spike count so we can allocate once instead of
        # building an intermediate list and concatenating a second time.
        total_spikes = sum(
            sum(len(a) for a in chunks) if chunks else 0
            for chunks in self.all_unit_spike_lists
        )
        spikes_1d = np.empty(total_spikes, dtype=np.float64)

        unit_indices = [0]
        pos = 0

        for unit_idx, unit_chunks in enumerate(self.all_unit_spike_lists):
            if unit_chunks:
                unit_arr = np.concatenate(unit_chunks)
                n = len(unit_arr)
                spikes_1d[pos : pos + n] = unit_arr
                pos += n
            unit_indices.append(pos)
            self.all_unit_spike_lists[unit_idx] = None  # free as we go

        # Convert spike times ms -> seconds before saving
        spikes_1d /= 1000.0

        # Save Main Output
        np.save(os.path.join(self.output_dir, "spikes.npy"), spikes_1d)

        # Save Metadata
        meta_data = {
            "modality": "spikes",
            "n_signals": self.num_units,
            "start_time": 0.0,
            "end_time": self.current_time_offset / 1000.0,
            "trial_id": self.trial_id,
            "sampling_rate": self.sampling_rate,
            "spike_indices": unit_indices,
            "stimuli_order": self.meta_segments,
            "smoothing": self.smooth_param,
        }

        with open(os.path.join(self.output_dir, "meta.yml"), "w") as f:
            yaml.dump(meta_data, f)

        print(
            f"Export Complete. Total Units: {self.num_units}, Total Time: {self.current_time_offset}ms"
        )
        print(f"Final Memory: {get_process_memory():.2f} MB")


def export_mozaik_trial_streamed(
    dsv_or_list,
    output_dir,
    trial_id,
    sampling_rate=1000.0,
    smooth_param=None,
    append_mode=False,
):
    """
    Wrapper function for backward compatibility.
    Processes the given DSV(s) in one go using the Exporter class.
    """
    exporter = MozaikTrialExporter(
        output_dir, trial_id, sampling_rate, smooth_param, append_mode
    )
    exporter.process_batch(dsv_or_list)
    exporter.finalize()


class MozaikScreenExporter:
    """
    Exports screen data (stimuli + timestamps) for a given trial by reading
    Mozaik DSV segment annotations (to locate source data files) and chunk
    JSONs (to determine stimulus order and modality).

    Generates a complete Experanto-compatible ``screen/`` directory with
    ``timestamps.npy``, per-trial ``meta/*.yml``, ``data/*.npy``, and
    ``combined_meta.json``.

    Parameters
    ----------
    output_dir : str
        Parent output directory (experiment root). A ``screen/`` subdirectory
        will be created inside it.
    chunk_paths : list of str
        Ordered list of chunk JSON file paths for this trial. Each JSON
        contains ``[{"modality": ..., "file": ..., "trial": ...}, ...]``.
    frame_duration_ms : float
        Simulation frame duration in ms (default 7.0).
    movie_frame_duration_ms : float
        Duration of a single video frame in ms (default 35.0).
    modality_filter : list of str or None
        If set, only export stimuli whose modality is in this list
        (e.g. ``["image"]``). Timestamps still advance for filtered-out
        stimuli to keep spike alignment.
    """

    def __init__(
        self,
        output_dir,
        chunk_paths,
        frame_duration_ms=7.0,
        movie_frame_duration_ms=35.0,
        modality_filter=None,
        tier_reference=None,
    ):
        self.screen_dir = os.path.join(output_dir, "screen")
        self.chunk_paths = chunk_paths
        self.frame_duration_ms = frame_duration_ms
        self.movie_frame_duration_ms = movie_frame_duration_ms
        self.modality_filter = modality_filter
        self._source_data_dir = None  # derived from DSV annotations
        self._source_meta_dir = None
        self._tier_reference = tier_reference  # condition_hash -> tier

        os.makedirs(os.path.join(self.screen_dir, "meta"), exist_ok=True)
        os.makedirs(os.path.join(self.screen_dir, "data"), exist_ok=True)

        # Device-level meta.yml
        with open(os.path.join(self.screen_dir, "meta.yml"), "w") as f:
            yaml.dump({"modality": "screen"}, f)

    def process_batch(self, dsv_or_list):
        """Extract the source data/meta directories from DSV segment annotations.

        Only the first segment with a valid ``movie_path`` is needed — all
        stimuli in a trial share the same source directory.
        """
        if self._source_data_dir is not None:
            return  # already resolved

        dsvs = dsv_or_list if isinstance(dsv_or_list, (list, tuple)) else [dsv_or_list]
        for dsv in dsvs:
            for seg in dsv.get_segments():
                if "stimulus" not in seg.annotations:
                    continue
                stim = seg.annotations["stimulus"]
                params = ast.literal_eval(stim) if isinstance(stim, str) else stim
                movie_path = params.get("movie_path")
                if movie_path:
                    self._source_data_dir = movie_path
                    # movie_path = <base_path>/screen/data
                    self._source_meta_dir = os.path.join(
                        os.path.dirname(movie_path.rstrip("/")), "meta"
                    )
                    print(f"Screen exporter: source data dir = {self._source_data_dir}")
                    print(f"Screen exporter: source meta dir = {self._source_meta_dir}")
                    return

    def finalize(self):
        """Walk chunk JSONs in order, build timestamps, copy files, write metadata.

        For each stimulus in the chunk JSONs, the output matches the ground-truth
        Experanto format:
        - **Video**: one entry with ``num_frames`` timestamps (one per frame).
        - **Image**: three separate entries — pre-blank (1 frame), image (1 frame),
          post-blank (1 frame) — mirroring the ``InternalStimulus`` /
          ``PixelMovieExperanto`` / ``InternalStimulus`` sequence in the simulation.
        - **Blank** (from chunk JSON): skipped (the simulation also skips these).
        """
        if self._source_data_dir is None:
            print("WARNING: No source data directory found. Skipping screen export.")
            return

        # Collect ordered stimuli from all chunks
        all_stimuli = []
        for chunk_path in self.chunk_paths:
            with open(chunk_path, "r") as f:
                all_stimuli.extend(json.load(f))

        timestamps_ms = []
        combined_meta = {}
        output_idx = 0
        fd = self.frame_duration_ms
        last_ts = 0.0  # running clock — tracks end of previous segment

        for item in all_stimuli:
            meta_file = item["file"]
            modality = item["modality"]

            # Blanks from the chunk JSON are skipped by the simulation
            if modality == "blank":
                continue

            # Load source meta for this stimulus
            src_meta = self._load_source_meta(meta_file)
            image_size = src_meta.get("image_size", [144, 256])

            if modality == "video":
                num_frames = src_meta["num_frames"]
                first_frame_idx = len(timestamps_ms)

                for _ in range(num_frames):
                    timestamps_ms.append(last_ts)
                    last_ts += self.movie_frame_duration_ms

                # Write video entry
                out_key = f"{output_idx:05d}"
                out_meta = dict(src_meta)
                out_meta["first_frame_idx"] = first_frame_idx
                out_meta["num_frames"] = num_frames
                combined_meta[out_key] = out_meta

                # Copy data file
                src_npy = os.path.join(
                    self._source_data_dir, meta_file.replace(".yml", ".npy")
                )
                dst_npy = os.path.join(self.screen_dir, "data", f"{out_key}.npy")
                if os.path.exists(src_npy):
                    shutil.copy2(src_npy, dst_npy)

                with open(
                    os.path.join(self.screen_dir, "meta", f"{out_key}.yml"), "w"
                ) as f:
                    yaml.dump(out_meta, f)
                output_idx += 1

            elif modality == "image":
                pre_blank_ms = fd * ((src_meta["pre_blank_period"] * 1000) // fd)
                presentation_ms = fd * ((src_meta["presentation_time"] * 1000) // fd)
                post_blank_ms = POST_BLANK_MS  # mirrors RandomizedExperanto (see constant at top)

                # --- Pre-blank entry ---
                out_key = f"{output_idx:05d}"
                pre_meta = {
                    "modality": "blank",
                    "first_frame_idx": len(timestamps_ms),
                    "num_frames": 1,
                    "image_size": image_size,
                    "interleave_value": 128.0,
                }
                timestamps_ms.append(last_ts)
                last_ts += pre_blank_ms
                combined_meta[out_key] = pre_meta
                with open(
                    os.path.join(self.screen_dir, "meta", f"{out_key}.yml"), "w"
                ) as f:
                    yaml.dump(pre_meta, f)
                output_idx += 1

                # --- Image entry ---
                out_key = f"{output_idx:05d}"
                img_meta = dict(src_meta)
                img_meta["first_frame_idx"] = len(timestamps_ms)
                img_meta["num_frames"] = 1
                timestamps_ms.append(last_ts)
                last_ts += presentation_ms
                combined_meta[out_key] = img_meta

                # Copy data file
                src_npy = os.path.join(
                    self._source_data_dir, meta_file.replace(".yml", ".npy")
                )
                dst_npy = os.path.join(self.screen_dir, "data", f"{out_key}.npy")
                if os.path.exists(src_npy):
                    shutil.copy2(src_npy, dst_npy)

                with open(
                    os.path.join(self.screen_dir, "meta", f"{out_key}.yml"), "w"
                ) as f:
                    yaml.dump(img_meta, f)
                output_idx += 1

                # --- Post-blank entry ---
                out_key = f"{output_idx:05d}"
                post_meta = {
                    "modality": "blank",
                    "first_frame_idx": len(timestamps_ms),
                    "num_frames": 1,
                    "image_size": image_size,
                    "interleave_value": 128.0,
                }
                timestamps_ms.append(last_ts)
                last_ts += post_blank_ms
                combined_meta[out_key] = post_meta
                with open(
                    os.path.join(self.screen_dir, "meta", f"{out_key}.yml"), "w"
                ) as f:
                    yaml.dump(post_meta, f)
                output_idx += 1

        # Trailing blank entry (matches ground-truth format)
        out_key = f"{output_idx:05d}"
        trailing_meta = {
            "modality": "blank",
            "first_frame_idx": len(timestamps_ms),
            "num_frames": 1,
            "image_size": image_size,
            "interleave_value": 128.0,
        }
        timestamps_ms.append(last_ts)
        combined_meta[out_key] = trailing_meta
        with open(os.path.join(self.screen_dir, "meta", f"{out_key}.yml"), "w") as f:
            yaml.dump(trailing_meta, f)
        output_idx += 1

        # Save timestamps (convert ms → seconds)
        timestamps = np.array(timestamps_ms, dtype=np.float64) / 1000.0
        np.save(os.path.join(self.screen_dir, "timestamps.npy"), timestamps)

        # Write combined_meta.json
        with open(os.path.join(self.screen_dir, "combined_meta.json"), "w") as f:
            json.dump(combined_meta, f)

        print(
            f"Screen export complete: {output_idx} entries, "
            f"{len(timestamps)} timestamp frames, "
            f"end_time={timestamps[-1]:.3f}s"
        )

    def _load_source_meta(self, meta_filename):
        """Load a per-stimulus YAML from the source screen/meta/ directory.

        If a tier_reference mapping was provided, overrides the tier field
        using the stimulus's condition_hash as the lookup key.
        """
        path = os.path.join(self._source_meta_dir, meta_filename)
        with open(path, "r") as f:
            meta = yaml.safe_load(f)
        if self._tier_reference is not None:
            condition_hash = meta.get("condition_hash")
            if condition_hash and condition_hash in self._tier_reference:
                meta["tier"] = self._tier_reference[condition_hash]
        return meta

    def _calc_stimulus_duration_ms(self, src_meta, fd):
        """Return the total time (ms) a stimulus occupies, without modifying any state."""
        modality = src_meta.get("modality", "blank")
        if modality == "video":
            return src_meta["num_frames"] * self.movie_frame_duration_ms
        elif modality == "image":
            pre_blank_ms = fd * ((src_meta["pre_blank_period"] * 1000) // fd)
            presentation_ms = fd * ((src_meta["presentation_time"] * 1000) // fd)
            post_blank_ms = POST_BLANK_MS  # mirrors RandomizedExperanto (see constant at top)
            return pre_blank_ms + presentation_ms + post_blank_ms
        else:  # blank
            return fd

    def _advance_timestamps(self, timestamps_ms, src_meta, fd, start_time_ms=None):
        """
        Append timestamps for one stimulus and return the number of frames
        added. Mirrors the simulation's RandomizedExperanto timing logic.

        For videos: one timestamp per frame, spaced by movie_frame_duration_ms.
        For images: pre_blank frame + image frame + post_blank frame, with
            durations discretised to multiples of frame_duration.
        For blanks: one frame with minimal duration.

        Parameters
        ----------
        start_time_ms : float or None
            If provided, use this as the base time instead of the last
            timestamp in ``timestamps_ms``.
        """
        modality = src_meta.get("modality", "blank")
        if start_time_ms is not None:
            last_ts = start_time_ms
        else:
            last_ts = timestamps_ms[-1] if timestamps_ms else 0.0
        frames_added = 0

        if modality == "video":
            num_frames = src_meta["num_frames"]
            for _ in range(num_frames):
                timestamps_ms.append(last_ts)
                last_ts += self.movie_frame_duration_ms
                frames_added += 1

        elif modality == "image":
            # Discretise to simulation frame duration
            pre_blank_ms = fd * ((src_meta["pre_blank_period"] * 1000) // fd)
            presentation_ms = fd * ((src_meta["presentation_time"] * 1000) // fd)

            # Pre-blank frame
            timestamps_ms.append(last_ts)
            last_ts += pre_blank_ms
            frames_added += 1

            # Image frame
            timestamps_ms.append(last_ts)
            last_ts += presentation_ms
            frames_added += 1

            # Post-blank frame (mirrors RandomizedExperanto; see POST_BLANK_MS at top)
            timestamps_ms.append(last_ts)
            last_ts += POST_BLANK_MS
            frames_added += 1

        elif modality == "blank":
            timestamps_ms.append(last_ts)
            last_ts += fd
            frames_added += 1

        return frames_added
