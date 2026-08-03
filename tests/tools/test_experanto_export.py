"""
Tests for the Experanto spike export (mozaik.tools.experanto_export.MozaikTrialExporter).

Uses synthetic, duck-typed DataStoreView / segment objects with KNOWN spike trains, so the
export can be driven and checked end-to-end without running a model. Verifies:

  a) the output format contract (meta.yml keys, CSR N+1 spike_indices, timeline end_time), and
  b) that the exported spikes are IDENTICAL to those "stored in the datastore" (the synthetic
     spike trains), with the documented per-segment offsetting and < duration windowing.

Also checks that the new stim_name_key / group_by_key parameters let the user select and name
stimuli by parameters other than movie_name / trial, with the defaults reproducing the old
behaviour.
"""

import os

import numpy as np
import yaml

from mozaik.tools.experanto_export import MozaikTrialExporter


class _Seg:
    """Minimal stand-in for a Mozaik/neo segment as the exporter consumes it."""

    def __init__(self, stim, trains=None):
        self.annotations = {"stimulus": stim}
        self._trains = trains or []

    def get_spiketrains(self):
        return self._trains

    def release(self):
        pass


class _DSV:
    def __init__(self, segments):
        self._segments = segments

    def get_segments(self):
        return self._segments


def _reconstruct(spikes, spike_indices):
    """Split the flat spikes array back into per-unit arrays via the CSR indices."""
    return [
        spikes[spike_indices[i] : spike_indices[i + 1]]
        for i in range(len(spike_indices) - 1)
    ]


def _load(output_dir):
    spikes = np.load(os.path.join(output_dir, "spikes.npy"))
    with open(os.path.join(output_dir, "meta.yml")) as f:
        meta = yaml.safe_load(f)
    return spikes, meta


def test_trial_export_format_and_spikes_identical(tmp_path):
    # trial 0: image A (dur 100), blank (dur 50), image C (dur 80); trial 1: image D (must be excluded)
    segs = [
        _Seg(
            {"trial": 0, "movie_name": "imgA", "duration": 100},
            [np.array([10.0, 50.0, 150.0]), np.array([20.0])],  # 150 >= dur -> dropped
        ),
        _Seg({"trial": 0, "duration": 50}),  # blank (no movie_name)
        _Seg(
            {"trial": 0, "movie_name": "imgC", "duration": 80},
            [np.array([5.0, 79.0]), np.array([])],
        ),
        _Seg(
            {"trial": 1, "movie_name": "imgD", "duration": 100},
            [np.array([1.0, 2.0, 3.0]), np.array([4.0])],  # different trial -> excluded
        ),
    ]

    out = str(tmp_path / "responses")
    exp = MozaikTrialExporter(out, trial_id=0, sampling_rate=1000.0)
    exp.process_batch([_DSV(segs)])
    exp.finalize()
    spikes, meta = _load(out)

    # --- format contract ---
    assert meta["modality"] == "spikes"
    assert meta["n_signals"] == 2
    assert meta["start_time"] == 0.0
    assert meta["sampling_rate"] == 1000.0
    assert meta["trial_id"] == 0
    assert meta["stimuli_order"] == ["imgA", "blank", "imgC"]
    idx = meta["spike_indices"]
    assert len(idx) == meta["n_signals"] + 1  # CSR N+1
    assert idx[0] == 0 and idx[-1] == len(spikes)
    assert all(idx[i] <= idx[i + 1] for i in range(len(idx) - 1))  # monotonic
    # end_time == cumulative duration of ALL trial-0 segments (incl. blank): 100+50+80 = 230 ms
    assert meta["end_time"] == 0.230

    # --- spikes identical to the datastore trains (offset per segment, windowed to < duration) ---
    # unit0: imgA [10,50] @off 0 ; imgC [5,79] @off 150  -> [10,50,155,229] ms
    # unit1: imgA [20] @off 0                             -> [20] ms
    u0, u1 = _reconstruct(spikes, idx)
    np.testing.assert_allclose(u0, np.array([10.0, 50.0, 155.0, 229.0]) / 1000.0)
    np.testing.assert_allclose(u1, np.array([20.0]) / 1000.0)


def test_custom_group_and_name_keys(tmp_path):
    # Group by a string "phase" (exercises the non-int equality path) and name by "label".
    segs = [
        _Seg(
            {"phase": "A", "label": "s1", "duration": 100},
            [np.array([10.0]), np.array([])],
        ),
        _Seg(
            {"phase": "B", "label": "s2", "duration": 100},
            [np.array([11.0]), np.array([])],
        ),
        _Seg(
            {"phase": "A", "label": "s3", "duration": 100},
            [np.array([12.0]), np.array([])],
        ),
    ]
    out = str(tmp_path / "responses")
    exp = MozaikTrialExporter(
        out,
        trial_id=0,
        group_by_key="phase",
        group_value="A",
        stim_name_key="label",
    )
    exp.process_batch([_DSV(segs)])
    exp.finalize()
    spikes, meta = _load(out)

    # only phase-A stimuli selected, named by "label"
    assert meta["stimuli_order"] == ["s1", "s3"]
    u0, _ = _reconstruct(spikes, meta["spike_indices"])
    # s1 [10] @off 0 ; s3 [12] @off 100 -> [10, 112] ms
    np.testing.assert_allclose(u0, np.array([10.0, 112.0]) / 1000.0)
