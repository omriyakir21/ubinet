import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np    
from sklearn.metrics import auc
import paths
import patch_to_score_MLP_utils as utils
import tensorflow as tf
from data_preparation.ScanNet.db_creation_scanNet_utils import load_as_pickle,save_as_pickle
from results.patch_to_score.patch_to_score_result_analysis import get_best_architecture_models_path

def train_models(models_folder_path,results_folder_path,data_for_training_folder_path,with_pesto):
    folds_training_dicts = load_as_pickle(os.path.join(data_for_training_folder_path,'folds_training_dicts.pkl'))
    os.makedirs(models_folder_path,exist_ok=True)
    os.makedirs(results_folder_path,exist_ok=True)
    model_log_dir = os.path.join(models_folder_path, 'logs')
    os.makedirs(model_log_dir,exist_ok=True)
    metric = 'pr_auc'
    max_pr_auc = 0
    grid_results = []
    n_layers = int(sys.argv[1])
    m_a = int(sys.argv[2])
    m_b_values = [1024]
    m_c = int(sys.argv[3])
    batch_size = 200
    n_early_stopping_epochs = 12
    for m_b in m_b_values:
        print(f'architecture:{n_layers}_{m_a}_{m_b}_{m_c}')
        architecture_models = []
        architecture_validation_predictions = []
        architecture_validation_labels = []
        architecture_test_predictions = []
        architecture_test_labels = []
        for i in range(len(folds_training_dicts)):
            print(f'fold {i}')
            tf.keras.backend.clear_session()
            model = utils.build_model_concat_size_and_n_patches_same_number_of_layers(m_a, m_b, m_c, n_layers,with_pesto)
            # Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                        loss='binary_crossentropy',
                        metrics=[tf.keras.metrics.AUC(curve='PR'), 'accuracy'])
            
            print(m_a, m_b, m_c, n_layers,
                n_early_stopping_epochs, batch_size, i)
            
            dict_for_training = folds_training_dicts[i]
            sizes_train = tf.squeeze(dict_for_training['sizes_train'],axis=1)
            components_train = dict_for_training['components_train']
            num_patches_train = tf.squeeze(dict_for_training['num_patches_train'],axis=1)
            uniprots_train = dict_for_training['uniprots_train']
            labels_train = dict_for_training['labels_train']
            sizes_validation = tf.squeeze(dict_for_training['sizes_validation'],axis=1)
            components_validation = dict_for_training['components_validation']
            num_patches_validation = tf.squeeze(dict_for_training['num_patches_validation'],axis=1)
            uniprots_validation = dict_for_training['uniprots_validation']
            labels_validation = dict_for_training['labels_validation']
            sizes_test = tf.squeeze(dict_for_training['sizes_test'],axis=1)
            components_test = dict_for_training['components_test']
            num_patches_test = tf.squeeze(dict_for_training['num_patches_test'],axis=1)
            uniprots_test = dict_for_training['uniprots_test']
            labels_test = dict_for_training['labels_test']
            class_weights = utils.compute_class_weight('balanced', classes=np.unique(labels_train.numpy()),
                                                    y=labels_train.numpy())
            # Convert class weights to a dictionary
            class_weight = {i: class_weights[i] for i in range(len(class_weights))}
            
            # Create a log directory for TensorBoard
            architecture_log_dir = os.path.join(model_log_dir, f"architecture_{n_layers}_{m_a}_{m_b}_{m_c}")
            os.makedirs(architecture_log_dir, exist_ok=True)
            fold_log_dir = os.path.join(architecture_log_dir, f"fold_{i}")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fold_log_dir, histogram_freq=1)

            loss_metric = 'val_auc'
            model.fit(
                [components_train, sizes_train, num_patches_train],
                labels_train,
                epochs=1,
                verbose=1,
                validation_data=(
                    [components_validation, sizes_validation, num_patches_validation], labels_validation),
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor=loss_metric,
                                                            mode='max',
                                                            patience=n_early_stopping_epochs,
                                                            restore_best_weights=True),
                                                            tensorboard_callback],
                batch_size=batch_size,
                class_weight=class_weight
            )

            yhat_validation = model.predict([components_validation, sizes_validation, num_patches_validation])
            architecture_validation_predictions.append(yhat_validation)
            architecture_validation_labels.append(labels_validation.numpy())
            
            
            yhat_test = model.predict([components_test, sizes_test, num_patches_test])
            architecture_test_predictions.append(yhat_test)
            architecture_test_labels.append(labels_test.numpy())

            architecture_models.append(model)

        all_architecture_labels = np.concatenate(architecture_validation_labels)
        all_architecture_predictions = np.concatenate(architecture_validation_predictions)
        precision, recall, _ = utils.precision_recall_curve(all_architecture_labels, all_architecture_predictions)
        pr_auc = auc(recall, precision)
        print(f'pr_auc is {pr_auc}')
        
        grid_results.append({'m_a': m_a, 'm_b': m_b, 'm_c': m_c, 'n_layers': n_layers,'batch_size':batch_size,'n_early_stopping_epochs':n_early_stopping_epochs, f'val_metric': pr_auc})
        if pr_auc > max_pr_auc:
            max_pr_auc = pr_auc
            best_models = architecture_models
            best_architecture = f'architecture:{n_layers}_{m_a}_{m_b}_{m_c}'
            best_architecture_test_predictions = architecture_test_predictions
            best_architecture_test_labels = architecture_test_labels
    
    models_architecture_folder = os.path.join(models_folder_path,best_architecture)
    os.makedirs(models_architecture_folder,exist_ok=True)
    results_architecture_folder = os.path.join(results_folder_path,best_architecture)
    os.makedirs(results_architecture_folder,exist_ok=True)
    utils.save_grid_search_results(grid_results,results_architecture_folder)
    
    for i in range(len(best_models)):
        best_models[i].save(os.path.join(models_architecture_folder, f'model{i}.keras'))

    utils.save_architecture_test_results(best_architecture_test_predictions,best_architecture_test_labels,results_architecture_folder)



def predict_over_test_set(models_dir_path,results_dir):
    architecture_dir_path = get_best_architecture_models_path(models_dir_path,results_dir)
    print(f'architecture_dir_path is {architecture_dir_path}')
    models = []
    for filename in os.listdir(architecture_dir_path):
        if filename.endswith('.keras'):
            models.append(tf.keras.models.load_model(os.path.join(architecture_dir_path, filename)))
    folds_training_dicts = load_as_pickle(os.path.join(paths.patch_to_score_data_for_training_path,
                                                                        'folds_traning_dicts.pkl'))
    all_predictions = []
    all_labels = []
    
    for i in range(len(folds_training_dicts)):
        dict_for_training = folds_training_dicts[i]
        sizes_test = tf.squeeze(dict_for_training['sizes_test'],axis=1)
        components_test = dict_for_training['components_test']
        num_patches_test = tf.squeeze(dict_for_training['num_patches_test'],axis=1)
        labels_test = dict_for_training['labels_test']
        model = models[i]
        predictions = model.predict([components_test, sizes_test, num_patches_test])
        all_predictions.append(predictions)
        all_labels.append(labels_test.numpy())
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)
    np.save(os.path.join(architecture_dir_path,'predictions_test.npy'),all_predictions)
    np.save(os.path.join(architecture_dir_path,'labels_test.npy'),all_labels)

    


if __name__ == "__main__":
    DATE = '03_04'
    with_pesto = False
    with_pesto_addition = '_with_pesto' if with_pesto else ''
    training_name = f'{DATE}{with_pesto_addition}'
    data_for_training_folder_path = os.path.join(paths.patch_to_score_data_for_training_path, f'{training_name}')
    models_folder_path = os.path.join(paths.patch_to_score_model_path, f'{training_name}')
    results_folder_path = os.path.join(paths.patch_to_score_results_path, f'{training_name}')
    
    train_models(models_folder_path,results_folder_path,data_for_training_folder_path,with_pesto)
    
    # predict_over_test_set(paths.with_MSA_50_plddt_0304_models_dir,paths.with_MSA_50_plddt_0304_results_dir)
    # print('done')
