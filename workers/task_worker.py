from PyQt5.QtCore import pyqtSignal
from workers.base_worker import BaseWorker

class TaskWorker(BaseWorker):
    finished = pyqtSignal(object)

    def __init__(self, func, *args):
        super().__init__()
        self.func = func
        self.args = args

    def run(self):
        result = self.func(*self.args)
        self.finished.emit(result)
        