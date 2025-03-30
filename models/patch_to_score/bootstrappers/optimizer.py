import tensorflow as tf


def bootstrap_adam(learning_rate: float) -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)

def bootstrap_sgd(learning_rate: float) -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.SGD(learning_rate=learning_rate)

def bootstrap_adamw(learning_rate: float) -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.AdamW(learning_rate=learning_rate)


optimizer_to_bootstrapper = {
    'sgd': bootstrap_sgd,
    'adam': bootstrap_adam,
    'adamw': bootstrap_adamw   
}

def build_optimizer_from_configuration(name: str, kwargs: dict) -> tf.keras.Model:
    supported_optimizers = list(optimizer_to_bootstrapper.keys())
    if name not in optimizer_to_bootstrapper.keys():
        raise Exception(f'model: {name} not supported. supported models: {supported_optimizers}')
    return optimizer_to_bootstrapper[name](**kwargs)
