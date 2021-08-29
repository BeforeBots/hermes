import tensorflow as tf
import pandas as pd


class FileCallback(tf.keras.callbacks.Callback):
    def __init__(self, param="", filepath=""):
        super(FileCallback, self).__init__()
        self.param = param

    def on_train_end(self, logs=None):
        s = f"""df.to_{self.param}()"""
        print(s)


obj = FileCallback.on_train_end()
