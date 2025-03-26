import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import uuid
import numpy as np    
from sklearn.metrics import auc
import paths
import utils
import tensorflow as tf
from data_preparation.ScanNet.db_creation_scanNet_utils import load_as_pickle
from models.patch_to_score.bootstrappers.loss import build_loss_from_configuration
from models.patch_to_score.bootstrappers.model import build_model_from_configuration
from models.patch_to_score.bootstrappers.optimizer import build_optimizer_from_configuration
from models.patch_to_score.dataset import PatchToScoreCrossValidationDataset, PatchToScoreDataset


def train(fold_index, dataset: PatchToScoreDataset,
          model: tf.keras.Model, optimizer: tf.keras.Optimizer, loss: tf.keras.Loss,
          fit_kwargs: dict,
          architecture_log_dir: str) -> tf.keras.Model:
    print(f'fold {fold_index}')
    tf.keras.backend.clear_session()
    
    # Compile the model
    model.compile(optimizer=optimizer,
                loss=loss,
                metrics=[tf.keras.metrics.AUC(curve='PR'), 'accuracy'])
    # print(f'architecture_dict: {architecture_dict}, fold {fold_index}')
           
    class_weights = utils.compute_class_weight('balanced', classes=np.unique(dataset.train_labels.numpy()),
                                                    y=dataset.train_labels.numpy())
    class_weight = {i: class_weights[i] for i in range(len(class_weights))}

    os.makedirs(architecture_log_dir, exist_ok=True)
    fold_log_dir = os.path.join(architecture_log_dir, f"fold_{fold_index}")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fold_log_dir, histogram_freq=1)
    
    loss_metric = 'val_auc'
    model.fit(
        [dataset.train_set, dataset.train_sizes, dataset.train_num_patch],
        dataset.train_labels.numpy(),
        epochs=fit_kwargs['epochs'],
        verbose=fit_kwargs['verbose'],
        validation_data=(
            [dataset.validation_set, dataset.validation_sizes, dataset.validation_num_patch],
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



def run_cross_validation(models_folder_path,results_folder_path,
                 model: tf.keras.Model, optimizer: tf.keras.Optimizer, loss: tf.keras.Loss,
                 model_kwargs: dict, fit_kwargs: dict, 
                 architecture_log_dir: str,
                 cross_validation_dataset: PatchToScoreCrossValidationDataset):
    grid_results = []
    architecture_models = []
    architecture_validation_predictions = []
    architecture_validation_labels = []
    architecture_test_predictions = []
    architecture_test_labels = []
    for i, dataset in enumerate(cross_validation_dataset.fold_datasets):
        model = train(i, dataset, model, optimizer, loss, fit_kwargs, architecture_log_dir)
        yhat_validation = model.predict([dataset.validation_set, dataset.validation_sizes, dataset.validation_num_patch])
        architecture_validation_predictions.append(yhat_validation)
        architecture_validation_labels.append(dataset.validation_labels.numpy())
        
        yhat_test = model.predict([dataset.test_set, dataset.test_sizes, dataset.test_num_patch])
        architecture_test_predictions.append(yhat_test)
        architecture_test_labels.append(dataset.test_labels.numpy())
        architecture_models.append(model)

    all_architecture_labels = np.concatenate(architecture_validation_labels)
    all_architecture_predictions = np.concatenate(architecture_validation_predictions)
    precision, recall, _ = utils.precision_recall_curve(all_architecture_labels, all_architecture_predictions)
    pr_auc = auc(recall, precision)
    print(f'pr_auc is {pr_auc}')

    architecture_dict = {**model_kwargs, **fit_kwargs}
    architecture_dict['val_metric'] = pr_auc
    grid_results.append(architecture_dict)
    
    architecture_name = f"architecture:{architecture_dict['n_layers']}_{architecture_dict['m_a']}_{architecture_dict['m_b']}_{architecture_dict['m_c']}"

    models_architecture_folder = os.path.join(models_folder_path,architecture_name)+'_second'
    os.makedirs(models_architecture_folder,exist_ok=True)
    results_architecture_folder = os.path.join(results_folder_path,architecture_name)+'_second'
    os.makedirs(results_architecture_folder,exist_ok=True)
    utils.save_grid_search_results(grid_results,results_architecture_folder)
    
    for i in range(len(architecture_models)):
        architecture_models[i].save(os.path.join(models_architecture_folder, f'model{i}.keras'))

    utils.save_architecture_test_results(architecture_test_predictions, architecture_test_labels,results_architecture_folder)


if __name__ == "__main__":
    hypothesis = sys.argv[1]
    experiments = os.listdir(f'configurations/data/{hypothesis}')
    for experiment in experiments:
        print('experiment:', experiment)

        with open(f'configurations/data/{hypothesis}/{experiment}', 'r') as f:
            train_configuration = json.load(f)

        DATE = '03_04'
        with_pesto = train_configuration['data']['use_pesto']
        # ABLATION STRING[i] = 1 MEANS THAT WE ARE USING THE I'TH FEATURE FROM THIS LIST OF FEATURES:
        # [patch size , scanNet_ubiq , scanNet_protein , pesto_protein , pesto_dna_rna , pesto_ion , pesto_ligand , pesto_lipid , average_plddt]
        ablation_string = train_configuration['data']['ablation_string']

        with_pesto_addition = f'_with_pesto' if with_pesto else ''
        training_name = f'{DATE}{with_pesto_addition}'
        print(training_name)
        training_name += f'_{ablation_string}'
        
        hypothesis_name = train_configuration['hypothesis']
        experiment_name = train_configuration['experiment']
        random_id = uuid.uuid4().hex[:10]
        
        models_folder_path = os.path.join(paths.patch_to_score_model_path, f'{training_name}')
        results_folder_path = os.path.join(paths.patch_to_score_results_path, 'hypotheses', hypothesis_name, experiment_name, random_id)
        model_log_dir = os.path.join(models_folder_path, 'logs')

        for path in [models_folder_path, results_folder_path, model_log_dir]:
            os.makedirs(path,exist_ok=True)

        with open(f'{results_folder_path}/configuration.json', 'w') as f:
            json.dump(train_configuration, f)

        print('models path:', models_folder_path)
        print('results path:', results_folder_path)

        model_configuration = train_configuration['model']
        
        model = build_model_from_configuration(**model_configuration)
        optimizer = build_optimizer_from_configuration(**train_configuration['compile']['optimizer'])
        loss = build_loss_from_configuration(**train_configuration['compile']['loss'])

        model_kwargs = model_configuration['kwargs']
        fit_kwargs = train_configuration['fit']
        compile_configuration = train_configuration['compile']
        architecture_log_dir = os.path.join(model_log_dir,
                        f"architecture_{model_kwargs['n_layers']}_{model_kwargs['m_a']}_{model_kwargs['m_b']}_{model_kwargs['m_c']}")
        
        cross_validation_dataset = PatchToScoreCrossValidationDataset(**train_configuration['data'])

        run_cross_validation(models_folder_path,results_folder_path,
                    model, optimizer, loss,
                    model_kwargs, fit_kwargs, 
                    architecture_log_dir,
                    cross_validation_dataset)
