import tensorflow as tf
import tensorflow.keras.backend as K

class LRS(tf.keras.callbacks.Callback):

    def __init__(self, interval = { 3 : 0.05 , 6 : 0.01 , 9 : 0.005 , 12 : 0.001}):
        super(LRS, self).__init__()
        self.interval = interval

    def on_epoch_begin(self, epoch, logs=None):

        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')

        lr = float(K.get_value(self.model.optimizer.learning_rate))

        if epoch in list(self.interval.keys()) :
            tf.keras.backend.set_value(self.model.optimizer.lr, self.interval[epoch])
            
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, self.model.optimizer.lr))
