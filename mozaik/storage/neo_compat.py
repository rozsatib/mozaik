r"""
Temporary compatibility helpers for loading datastores pickled with older Neo.

TODO: Remove this module once pre-neo==0.14.0 datastore pickles no longer need
to be supported. When bumping Neo for good, also update README.rst and the
GitHub Actions install command.
"""

from contextlib import contextmanager

from neo.core.analogsignal import AnalogSignal
from neo.core.epoch import Epoch
from neo.core.event import Event
from neo.core.irregularlysampledsignal import IrregularlySampledSignal
from neo.core.objectlist import ObjectList
from neo.core.segment import Segment
from neo.core.spiketrain import SpikeTrain

try:
    from neo.core.spiketrainlist import SpikeTrainList
except ImportError:
    SpikeTrainList = None


def _make_object_list(cls, parent):
    if cls is SpikeTrain and SpikeTrainList is not None:
        return SpikeTrainList(parent=parent)
    return ObjectList(cls, parent=parent)


def _object_list_types():
    if SpikeTrainList is None:
        return ObjectList
    return (ObjectList, SpikeTrainList)


def _coerce_object_list(parent, private, public, cls):
    d = parent.__dict__

    if public in d and private not in d:
        items = d.pop(public)
    elif private in d and not isinstance(d[private], _object_list_types()):
        items = d[private]
    else:
        if private not in d:
            d[private] = _make_object_list(cls, parent)
        return

    d[private] = _make_object_list(cls, parent)
    for item in items:
        d[private].append(item)


def _copy_legacy_annotation(annotations, current_key, legacy_key):
    if current_key not in annotations and legacy_key in annotations:
        annotations[current_key] = annotations[legacy_key]


def _fix_legacy_recording_annotations(segment):
    for spiketrain in segment.spiketrains:
        _copy_legacy_annotation(spiketrain.annotations, "channel_id", "source_id")

    for analogsignal in segment.analogsignals:
        _copy_legacy_annotation(analogsignal.annotations, "channel_ids", "source_ids")


def fix_legacy_block(block):
    _coerce_object_list(block, "_segments", "segments", Segment)
    for segment in block.segments:
        segment.block = block
    return block


def fix_legacy_segment(segment):
    containers = (
        ("_spiketrains", "spiketrains", SpikeTrain),
        ("_analogsignals", "analogsignals", AnalogSignal),
        ("_events", "events", Event),
        ("_epochs", "epochs", Epoch),
        ("_irregularlysampledsignals", "irregularlysampledsignals", IrregularlySampledSignal),
    )

    for private, public, cls in containers:
        _coerce_object_list(segment, private, public, cls)
        for item in segment.__dict__[private]:
            try:
                item.segment = segment
            except Exception:
                pass

    _fix_legacy_recording_annotations(segment)
    return segment


@contextmanager
def patch_legacy_neo_copy():
    targets = [SpikeTrain, AnalogSignal, Event, Epoch, IrregularlySampledSignal]
    original_news = {cls: cls.__new__ for cls in targets}

    def make_patched_new(orig_new):
        def patched_new(cls, *args, **kwargs):
            try:
                return orig_new(cls, *args, **kwargs)
            except ValueError as e:
                if "copy" not in str(e).lower():
                    raise
                if "copy" in kwargs:
                    kwargs["copy"] = None
                safe_args = tuple(None if isinstance(x, bool) else x for x in args)
                return orig_new(cls, *safe_args, **kwargs)

        return patched_new

    for cls in targets:
        cls.__new__ = make_patched_new(original_news[cls])

    try:
        yield
    finally:
        for cls in targets:
            cls.__new__ = original_news[cls]
