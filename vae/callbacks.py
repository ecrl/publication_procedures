r"""Training callback objects/functions"""
import sys


class CallbackOperator(object):
    """
    CallbackOperator: executes individual callback steps at each step
    """

    def __init__(self):

        self.cb = []

    def add_cb(self, cb):

        self.cb.append(cb)

    def on_train_begin(self):

        for cb in self.cb:
            if not cb.on_train_begin():
                return False
        return True

    def on_train_end(self):

        for cb in self.cb:
            if not cb.on_train_end():
                return False
        return True

    def on_epoch_begin(self, epoch):

        for cb in self.cb:
            if not cb.on_epoch_begin(epoch):
                return False
        return True

    def on_epoch_end(self, epoch):

        for cb in self.cb:
            if not cb.on_epoch_end(epoch):
                return False
        return True

    def on_batch_begin(self, batch):

        for cb in self.cb:
            if not cb.on_batch_begin(batch):
                return False
        return True

    def on_batch_end(self, batch):

        for cb in self.cb:
            if not cb.on_batch_end(batch):
                return False
        return True

    def on_loss_begin(self, batch):

        for cb in self.cb:
            if not cb.on_loss_begin(batch):
                return False
        return True

    def on_loss_end(self, batch):

        for cb in self.cb:
            if not cb.on_loss_end(batch):
                return False
        return True

    def on_step_begin(self, batch):

        for cb in self.cb:
            if not cb.on_step_begin(batch):
                return False
        return True

    def on_step_end(self, batch):

        for cb in self.cb:
            if not cb.on_step_end(batch):
                return False
        return True


class Callback(object):
    """
    Base Callback object
    """

    def __init__(self): pass
    def on_train_begin(self): return True
    def on_train_end(self): return True
    def on_epoch_begin(self, epoch): return True
    def on_epoch_end(self, epoch): return True
    def on_batch_begin(self, batch): return True
    def on_batch_end(self, batch): return True
    def on_loss_begin(self, batch): return True
    def on_loss_end(self, batch): return True
    def on_step_begin(self, batch): return True
    def on_step_end(self, batch): return True
