import tensorflow as tf


def bootstrap_adam(learning_rate: float) -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)

def bootstrap_sgd(learning_rate: float) -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.SGD(learning_rate=learning_rate)

def bootstrap_adamw(learning_rate: float) -> tf.keras.optimizers.Optimizer:
    # return tf.keras.optimizers.AdamW(learning_rate=learning_rate)  # only supported on tensorflow=2.16.1
    return tf.optimizers.Adam(learning_rate=learning_rate, weight_decay=0.004)



optimizer_to_bootstrapper = {
    'sgd': bootstrap_sgd,
    'adam': bootstrap_adam,
    'adamw': bootstrap_adamw  # TODO : remove this option, and make it a configuration of adam   
}

def build_optimizer_from_configuration(name: str, kwargs: dict) -> tf.keras.optimizers.Optimizer:
    supported_optimizers = list(optimizer_to_bootstrapper.keys())
    if name not in optimizer_to_bootstrapper.keys():
        raise Exception(f'optimizer: {name} not supported. supported optimizers: {supported_optimizers}')
    return optimizer_to_bootstrapper[name](**kwargs)
