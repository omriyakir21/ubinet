import uuid
import json
import os
import sys

from typing import Tuple, List
from argparse import Namespace, ArgumentParser

import numpy as np
import tensorflow as tf
from sklearn.metrics import auc

from models.patch_to_score.dataset import PatchToScoreCrossValidationDataset, PatchToScoreDataset
from models.patch_to_score.bootstrappers.optimizer import build_optimizer_from_configuration
from models.patch_to_score.bootstrappers.model import build_model_from_configuration
from models.patch_to_score.bootstrappers.loss import build_loss_from_configuration
from models.patch_to_score import utils
import paths


def train(fold_index, dataset: PatchToScoreDataset,
          model: tf.keras.Model, optimizer: tf.keras.Optimizer, loss: tf.keras.Loss,
          fit_kwargs: dict,
          results_folder_path: str) -> tf.keras.Model:
    print(f'fold {fold_index}')
    tf.keras.backend.clear_session()

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[tf.keras.metrics.AUC(curve='PR'), 'accuracy'])

    class_weights = utils.compute_class_weight('balanced', classes=np.unique(dataset.train_labels.numpy()),
                                               y=dataset.train_labels.numpy())
    class_weight = {i: class_weights[i] for i in range(len(class_weights))}

    fold_log_dir = os.path.join(
        results_folder_path, 'logs', f'fold_{fold_index}')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fold_log_dir, histogram_freq=1)

    loss_metric = 'val_auc'
    model.fit(
        [dataset.train_set, dataset.train_sizes, dataset.train_num_patch],
        dataset.train_labels.numpy(),
        epochs=fit_kwargs['epochs'],
        verbose=fit_kwargs['verbose'],
        validation_data=(
            [dataset.validation_set, dataset.validation_sizes,
                dataset.validation_num_patch],
            dataset.validation_labels.numpy()),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor=loss_metric,
                                                    mode='max',
                                                    patience=fit_kwargs['n_early_stopping_epochs'],
                                                    restore_best_weights=True),
                   tensorboard_callback],
        batch_size=fit_kwargs['batch_size'],
        class_weight=class_weight
    )
    return model


def bootstrap_train(train_configuration: dict) -> Tuple[tf.keras.Model, tf.keras.Optimizer, tf.keras.Loss]:
    model = build_model_from_configuration(**train_configuration['model'])
    optimizer = build_optimizer_from_configuration(
        **train_configuration['compile']['optimizer'])
    loss = build_loss_from_configuration(
        **train_configuration['compile']['loss'])
    return model, optimizer, loss


def run_cross_validation(results_folder_path: str,
                         train_configuration: dict,
                         model_kwargs: dict,
                         cross_validation_dataset: PatchToScoreCrossValidationDataset) -> None:
    grid_results = []
    architecture_models = []
    architecture_validation_predictions = []
    architecture_validation_labels = []
    architecture_test_predictions = []
    architecture_test_labels = []
    for i, dataset in enumerate(cross_validation_dataset.fold_datasets):
        model, optimizer, loss = bootstrap_train(train_configuration)
        model = train(i, dataset, model, optimizer, loss,
                      train_configuration['fit'], results_folder_path)
        yhat_validation = model.predict(
            [dataset.validation_set, dataset.validation_sizes, dataset.validation_num_patch])
        architecture_validation_predictions.append(yhat_validation)
        architecture_validation_labels.append(
            dataset.validation_labels.numpy())

        yhat_test = model.predict(
            [dataset.test_set, dataset.test_sizes, dataset.test_num_patch])
        architecture_test_predictions.append(yhat_test)
        architecture_test_labels.append(dataset.test_labels.numpy())
        architecture_models.append(model)

    all_architecture_labels = np.concatenate(architecture_validation_labels)
    all_architecture_predictions = np.concatenate(
        architecture_validation_predictions)
    precision, recall, _ = utils.precision_recall_curve(
        all_architecture_labels, all_architecture_predictions)
    pr_auc = auc(recall, precision)
    print(f'pr_auc is {pr_auc}')

    architecture_dict = {**model_kwargs, **train_configuration['fit']}
    architecture_dict['val_metric'] = pr_auc
    grid_results.append(architecture_dict)

    utils.save_grid_search_results(grid_results, results_folder_path)

    for i in range(len(architecture_models)):
        architecture_models[i].save(os.path.join(
            results_folder_path, f'model{i}.keras'))

    utils.save_architecture_test_results(
        architecture_test_predictions, architecture_test_labels, results_folder_path)


def load_configuration() -> dict:
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", dest="configuration_path",
                        help="path to configuration file")
    args = parser.parse_args()
    configuration_path = args.configuration_path
    print('experiment configuration file:', configuration_path)
    with open(configuration_path, 'r') as f:
        train_configuration = json.load(f)
    return train_configuration


if __name__ == "__main__":
    train_configuration = load_configuration()
    hypothesis_name = train_configuration['hypothesis']
    experiment_name = train_configuration['experiment']
    random_id = uuid.uuid4().hex[:10]
    
    results_folder_path = os.path.join(
        paths.patch_to_score_results_path, 'hypotheses', hypothesis_name, experiment_name, random_id)
    os.makedirs(results_folder_path, exist_ok=True)
    print('results path:', results_folder_path)
    
    with open(f'{results_folder_path}/configuration.json', 'w') as f:
        json.dump(train_configuration, f)

    model_kwargs = train_configuration['model']['kwargs']
    cross_validation_dataset = PatchToScoreCrossValidationDataset(
        **train_configuration['data'])

    run_cross_validation(results_folder_path,
                         train_configuration,
                         model_kwargs,
                         cross_validation_dataset)
