# import tensorflow as tf
from keras import ops
import keras
from utime.evaluation.utils import wrapper


def _get_shapes_and_one_hot(y_true, y_pred):
    shape = y_pred.get_shape()
    n_classes = shape[-1]
    # Squeeze dim -1 if it is == 1, otherwise leave it
    dims = ops.cond(ops.equal(y_true.shape[-1] or -1, 1), lambda: ops.shape(y_true)[:-1], lambda: ops.shape(y_true))
    y_true = ops.reshape(y_true, dims)
    y_true = ops.one_hot(ops.cast(y_true, 'uint8'), depth=n_classes)
    return y_true, shape, n_classes


def sparse_dice_loss(y_true, y_pred, smooth=1):
    """
    Approximates the class-wise dice coefficient computed per-batch element
    across spatial image dimensions. Returns the 1 - mean(per_class_dice) for
    each batch element.
    :param y_true:
    :param y_pred:
    :param smooth:
    :return:
    """
    y_true, shape, n_classes = _get_shapes_and_one_hot(y_true, y_pred)
    reduction_dims = range(len(shape))[1:-1]

    intersection = ops.reduce_sum(y_true * y_pred, axis=reduction_dims)
    union = ops.reduce_sum(y_true + y_pred, axis=reduction_dims)
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1.0 - ops.reduce_mean(dice, axis=-1, keepdims=True)


class SparseDiceLoss(keras.losses.Loss):
    """ tf reduction wrapper for sparse_dice_loss """
    def __init__(self,
                 reduction,
                 smooth=1,
                 name='sparse_dice_loss',
                 **kwargs):
        self.smooth = smooth
        super(SparseDiceLoss, self).__init__(
            name=name,
            reduction=reduction
        )

    def get_config(self):
        config = super().get_config()
        config.update({'smooth': self.smooth})
        return config

    def call(self, y_true, y_pred):
        return sparse_dice_loss(y_true, y_pred, smooth=self.smooth)

class IgnoreOutOfBoundSparseCategoricalCrossEntropy(keras.losses.SparseCategoricalCrossentropy):
    
    def __init__(self, from_logits=False, ignore_class=None, reduction="sum_over_batch_size", axis=-1, name="sparse_categorical_crossentropy", dtype=None):
        super().__init__(from_logits, ignore_class, reduction, axis, name, dtype)
    
    def call(self, y_true, y_pred):
        return wrapper(super().call, y_true, y_pred)
    
class IgnoreOutOfBoundSparseCategoricalAccuracy(keras.metrics.MeanMetricWrapper):
    def __init__(self, name="sparse_categorical_accuracy_ignore", dtype=None):
        super().__init__(fn=lambda y_true, y_pred: wrapper(keras.metrics.sparse_categorical_accuracy, y_true, y_pred), name=name, dtype=dtype)
        # Metric should be maximized during optimization.
        self._direction = "up"

    def get_config(self):
        return {"name": self.name, "dtype": self.dtype}