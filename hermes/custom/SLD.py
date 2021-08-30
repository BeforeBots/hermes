import tensorflow as tf
import requests
import numpy as np
import json
from collections import defaultdict


class SLD(tf.keras.callbacks.Callback):

    def __init__(self, port="9000", path="",
                 field="data",
                 headers=None,
                 send_as_json=True,
                 mode=["on_epoch_end"]):

        super(SLD, self).__init__()

        self.root = "http://localhost:" + port
        self.path = path
        self.field = field
        self.headers = headers
        self.send_as_json = send_as_json
        self.mode = mode
        self.getmap = defaultdict(bool)

        for i in self.mode:
            self.getmap[i + "_val"] = True

    def on_train_begin(self, logs=None):
        if self.getmap["on_train_begin_val"] == True:
            self.helper(logs=logs, message="on train begin")

    def on_train_end(self, logs=None):
        if self.getmap["on_train_end_val"] == True:
            self.helper(logs=logs, message="on train end")

    def on_epoch_begin(self, epoch, logs=None):
        if self.getmap["on_epoch_begin_val"] == True:
            self.helper(epoch, logs=logs, message="on epoch begin")

    def on_epoch_end(self, epoch, logs=None):
        if self.getmap["on_epoch_end_val"] == True:
            self.helper(epoch, logs=logs, message="on epoch end")

    def on_test_begin(self, logs=None):
        if self.getmap["on_test_begin_val"] == True:
            self.helper(logs=logs, message="on test begin")

    def on_test_end(self, logs=None):
        if self.getmap["on_test_end_val"] == True:
            self.helper(logs=logs, message="on test end")

    def on_predict_begin(self, logs=None):
        if self.getmap["on_predict_begin_val"] == True:
            self.helper(logs=logs, message="on predict begin")

    def on_predict_end(self, logs=None):
        if self.getmap["on_predict_end_val"] == True:
            self.helper(logs=logs, message="on predict end")

    def on_train_batch_begin(self, batch, logs=None):
        if self.getmap["on_train_batch_begin_val"] == True:
            self.helper(batch_no=batch, logs=logs,
                        message="on train batch begin")

    def on_train_batch_end(self, batch, logs=None):
        if self.getmap["on_train_batch_end_val"] == True:
            self.helper(batch_no=batch, logs=logs,
                        message="on train batch end")

    def on_test_batch_begin(self, batch, logs=None):
        if self.getmap["on_test_batch_begin_val"] == True:
            self.helper(batch_no=batch, logs=logs,
                        message="on test batch begin")

    def on_test_batch_end(self, batch, logs=None):
        if self.getmap["on_test_batch_end_val"] == True:
            self.helper(batch_no=batch, logs=logs,
                        message="on test batch end")

    def on_predict_batch_begin(self, batch, logs=None):
        if self.getmap["on_predict_batch_begin_val"] == True:
            self.helper(batch_no=batch, logs=logs,
                        message="on predict batch begin")

    def on_predict_batch_end(self, batch, logs=None):
        if self.getmap["on_predict_batch_end_val"] == True:
            self.helper(batch_no=batch, logs=logs,
                        message="on predict batch end")

    def helper(self, epoch_no=None, batch_no=None, logs=None, message=None):

        if requests is None:
            raise ImportError('RemoteMonitor requires the `requests` library.')

        logs = logs or {}
        send = {}

        if epoch_no != None:
            send['epoch_no'] = epoch_no
        elif batch_no != None:
            send['batch_no'] = batch_no
        elif message != None:
            send['message'] = message

        for k, v in logs.items():
            if isinstance(v, (np.ndarray, np.generic)):
                send[k] = v.item()
            else:
                send[k] = v
        try:
            if self.send_as_json:
                requests.post(self.root + self.path,
                              json=send, headers=self.headers)
            else:
                requests.post(
                    self.root + self.path, {self.field: json.dumps(send)},
                    headers=self.headers)
        except requests.exceptions.RequestException:
            print('Warning: could not reach RemoteMonitor '
                  'root server at ' + str(self.root))
