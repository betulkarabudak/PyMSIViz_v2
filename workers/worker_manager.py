from PyQt5.QtCore import QObject
class WorkerManager(QObject):
    def __init__(self):
        super().__init__()
        self._current_worker = None

    def start(self, worker):
        self.cancel()
        self._current_worker = worker

        # Worker lifecycle signals
        if hasattr(worker, "result"):
            worker.result.connect(self._on_done)
        if hasattr(worker, "cancelled"):
            worker.cancelled.connect(self._on_done)
        if hasattr(worker, "error"):
            worker.error.connect(self._on_error)

        worker.start()

    def _on_done(self, *args):
        self._cleanup()

    def _on_error(self, msg):
        self._cleanup()

    def cancel(self):
        if self._current_worker is not None:
            try:
                self._current_worker.cancel()
            except Exception:
                pass

    def _cleanup(self):
        if self._current_worker is not None:
            try:
                self._current_worker.quit()
                self._current_worker.wait(100)
            except Exception:
                pass
            self._current_worker = None