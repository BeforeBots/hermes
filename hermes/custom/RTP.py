import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 

class TrainingPlot(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.fig = plt.figure(figsize=(6,4))
        self.ax = self.fig.add_subplot()
        self.ax.set_xlabel('Epoch #')
        self.ax.set_ylabel('loss')
    
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs['loss'])
        
        epochs = np.arange(0, len(self.losses))
        self.ax.plot(epochs, self.losses, "b-")
        self.fig.canvas.draw()
