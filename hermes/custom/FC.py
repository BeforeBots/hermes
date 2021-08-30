import tensorflow as tf
import pandas as pd
from collections import defaultdict


class FileCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_format="", filepath=""):
        super(FileCallback, self).__init__()
        self.save_format = save_format
        self.dfparams = defaultdict(list)
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        for i in keys:
            self.dfparams[i].append(logs[i])

    def on_train_end(self, logs=None):
        df = pd.DataFrame(self.dfparams)
        s = f"""df.to_{self.save_format}('hardcoded.csv')"""
        eval(s)
