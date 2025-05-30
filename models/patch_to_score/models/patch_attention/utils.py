import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import GlobalAveragePooling1D


@tf.keras.utils.register_keras_serializable(package="MyCustomLayersEncoderMLP")
class GlobalSumPooling(GlobalAveragePooling1D):
    def __init__(self, data_format='channels_last', keepdims=False, **kwargs):
        super(GlobalSumPooling, self).__init__(
            data_format=data_format, **kwargs)
        self.keepdims = keepdims

    def call(self, inputs, mask=None):
        steps_axis = 1 if self.data_format == "channels_last" else 2
        if mask is not None:
            mask = tf.cast(mask, inputs[0].dtype)
            mask = tf.expand_dims(
                mask, 2 if self.data_format == "channels_last" else 1
            )
            inputs *= mask
            return backend.sum(
                inputs, axis=steps_axis, keepdims=self.keepdims
            )

    def get_config(self):
        config = super(GlobalSumPooling, self).get_config()
        config.update({
            "data_format": self.data_format,
            "keepdims": self.keepdims
        })
        return config


def saveAsPickle(object, fileName):
    with open(fileName + '.pkl', 'wb') as file:
        pickle.dump(object, file)


def loadPickle(fileName):
    with open(fileName, 'rb') as file:
        object = pickle.load(file)
        return object


def plot_precision_recall(y_probs, labels, header, save_path):
    precision, recall, _ = precision_recall_curve(labels, y_probs)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    pr_auc = round(pr_auc, 3)
    plt.plot(recall, precision,
             label='Precision-Recall Curve, AUC =' + str(pr_auc))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(header)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def k_computation(prediction, training_ub_ratio, with_class_weights=False):
    if with_class_weights:
        training_ub_ratio = 0.5
    val = 1 - prediction
    if val == 0:
        return
    K = ((1 - training_ub_ratio) * prediction) / ((training_ub_ratio) * (val))
    return K


def prediction_function_using_bayes_factor_computation(priorUb, KValue):
    finalPrediction = float((KValue * priorUb) /
                            ((KValue * priorUb) + (1 - priorUb)))
    return finalPrediction
