from PyQt5.QtCore import QThread, pyqtSignal
import traceback
class CancelableTaskWorker(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()
    def __init__(self, task_fn, *args, **kwargs):
        super().__init__()
        self._task_fn = task_fn
        self._args = args
        self._kwargs = kwargs
        self._cancel = False  

    def cancel(self):
        self._cancel = True

    def is_cancelled(self):
        return self._cancel

    def run(self):
        try:
            payload = self._task_fn(
                *self._args,
                progress_cb=self._emit_progress,
                cancel_cb=self.is_cancelled,
                **self._kwargs)

            if self._cancel:
                self.cancelled.emit()
                return

            if payload is None:
                self.cancelled.emit()
                return

            self.result.emit(payload)

        except Exception:
            if self._cancel:
                self.cancelled.emit()
            else:
                self.error.emit(traceback.format_exc())

    def _emit_progress(self, value):
        if self._cancel:
            return
        try:
            self.progress.emit(int(value))
        except Exception:
            pass
