import tensorflow as tf

def scheduler(epoch, lr):
    if epoch < 2:
        print("LR => \n",lr)
        return lr
    else:
        print("LR now => \n",lr * tf.math.exp(-0.1))
        return lr * tf.math.exp(-0.1)


def LRS():
    return tf.keras.callbacks.LearningRateScheduler(scheduler)
