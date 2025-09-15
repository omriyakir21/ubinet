from typing import Union
import tensorflow as tf


def bootstrap_bce() -> Union[tf.keras.losses.Loss, str]:
    return 'binary_crossentropy'  # TODO : switch to Loss object
    # return tf.keras.losses.BinaryCrossentropy()

loss_to_bootstrapper = {
    'binary_cross_entropy': bootstrap_bce
}

def build_loss_from_configuration(name: str, kwargs: dict) -> tf.keras.losses.Loss:
    supported_losses = list(loss_to_bootstrapper.keys())
    if name not in loss_to_bootstrapper.keys():
        raise Exception(f'loss: {name} not supported. supported losses: {supported_losses}')
    return loss_to_bootstrapper[name](**kwargs)