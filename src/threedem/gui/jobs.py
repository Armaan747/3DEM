"""Run slow work (parsing, voxelizing, solving) off the UI thread."""

from __future__ import annotations

import traceback
import warnings

from PySide6 import QtCore


class _Signals(QtCore.QObject):
    done = QtCore.Signal(object, object)  # result, list of warning messages
    failed = QtCore.Signal(str)


class Job(QtCore.QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = _Signals()

    def run(self):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = self.fn()
            self.signals.done.emit(result, [str(w.message) for w in caught])
        except Exception as exc:  # shown to the user in the UI
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.signals.failed.emit(detail)


_live: set[Job] = set()


def run_job(fn, on_done, on_failed) -> None:
    """Run ``fn()`` in the thread pool; ``on_done(result, warnings)`` / ``on_failed(text)`` fire on the UI thread."""
    job = Job(fn)
    _live.add(job)

    def finish(*args, cb):
        _live.discard(job)
        cb(*args)

    job.signals.done.connect(lambda r, w: finish(r, w, cb=on_done))
    job.signals.failed.connect(lambda t: finish(t, cb=on_failed))
    QtCore.QThreadPool.globalInstance().start(job)
