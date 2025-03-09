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

def train_fold(architecture_dict,fold_index,fold_dict,model_log_dir,with_pesto,ablation_string):
    print(f'fold {fold_index}')
    tf.keras.backend.clear_session()
    model = utils.build_model_concat_size_and_n_patches_same_number_of_layers(architecture_dict,with_pesto,ablation_string)
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss='binary_crossentropy',
                metrics=[tf.keras.metrics.AUC(curve='PR'), 'accuracy'])
    print(f'architecture_dict: {architecture_dict}, fold {fold_index}')
           
    class_weights = utils.compute_class_weight('balanced', classes=np.unique(fold_dict['labels_train'].numpy()),
                                                    y=fold_dict['labels_train'].numpy())
    class_weight = {i: class_weights[i] for i in range(len(class_weights))}
    architecture_log_dir = os.path.join(model_log_dir,
                    f"architecture_{architecture_dict['n_layers']}_{architecture_dict['m_a']}_{architecture_dict['m_b']}_{architecture_dict['m_c']}")
    os.makedirs(architecture_log_dir, exist_ok=True)
    fold_log_dir = os.path.join(architecture_log_dir, f"fold_{fold_index}")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fold_log_dir, histogram_freq=1)
    
    loss_metric = 'val_auc'
    model.fit(
        [fold_dict['components_train'], fold_dict['sizes_train'], fold_dict['num_patches_train']],
        fold_dict['labels_train'].numpy(),
        epochs=200,
        verbose=1,
        validation_data=(
            [fold_dict['components_validation'], fold_dict['sizes_validation'], fold_dict['num_patches_validation']],
              fold_dict['labels_validation'].numpy()),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor=loss_metric,
                                                    mode='max',
                                                    patience=architecture_dict['n_early_stopping_epochs'],
                                                    restore_best_weights=True),
                                                    tensorboard_callback],
        batch_size=architecture_dict['batch_size'],
        class_weight=class_weight
    )
    return model



def train_models(models_folder_path,results_folder_path,data_for_training_folder_path,with_pesto,ablation_string):
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
        architecture_dict = {'m_a': m_a, 'm_b': m_b, 'm_c': m_c, 'n_layers': n_layers,'batch_size':batch_size,'n_early_stopping_epochs':n_early_stopping_epochs}
        architecture_models = []
        architecture_validation_predictions = []
        architecture_validation_labels = []
        architecture_test_predictions = []
        architecture_test_labels = []
        for i in range(len(folds_training_dicts)):
            # print(f'fold {i}')
            # tf.keras.backend.clear_session()
            # model = utils.build_model_concat_size_and_n_patches_same_number_of_layers(m_a, m_b, m_c, n_layers,with_pesto,ablation_string)
            # # Compile the model
            # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            #             loss='binary_crossentropy',
            #             metrics=[tf.keras.metrics.AUC(curve='PR'), 'accuracy'])
            
            # print(m_a, m_b, m_c, n_layers,
            #     n_early_stopping_epochs, batch_size, i)
            
            # dict_for_training = folds_training_dicts[i]
            # sizes_train = tf.squeeze(dict_for_training['sizes_train'],axis=1)
            # components_train = dict_for_training['components_train']
            # num_patches_train = tf.squeeze(dict_for_training['num_patches_train'],axis=1)
            # labels_train = dict_for_training['labels_train']
            # sizes_validation = tf.squeeze(dict_for_training['sizes_validation'],axis=1)
            # components_validation = dict_for_training['components_validation']
            # num_patches_validation = tf.squeeze(dict_for_training['num_patches_validation'],axis=1)
            # labels_validation = dict_for_training['labels_validation']
            # sizes_test = tf.squeeze(dict_for_training['sizes_test'],axis=1)
            # components_test = dict_for_training['components_test']
            # num_patches_test = tf.squeeze(dict_for_training['num_patches_test'],axis=1)
            # labels_test = dict_for_training['labels_test']

            # if with_pesto:
            #     components_train, components_validation, components_test = utils.filter_with_ablation_string(ablation_string,components_train,components_validation,components_test)            
            # class_weights = utils.compute_class_weight('balanced', classes=np.unique(labels_train.numpy()),
            #                                         y=labels_train.numpy())
            # # Convert class weights to a dictionary
            # class_weight = {i: class_weights[i] for i in range(len(class_weights))}
            
            # # Create a log directory for TensorBoard
            # architecture_log_dir = os.path.join(model_log_dir, f"architecture_{n_layers}_{m_a}_{m_b}_{m_c}")
            # os.makedirs(architecture_log_dir, exist_ok=True)
            # fold_log_dir = os.path.join(architecture_log_dir, f"fold_{i}")
            # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fold_log_dir, histogram_freq=1)

            # loss_metric = 'val_auc'
            # model.fit(
            #     [components_train, sizes_train, num_patches_train],
            #     labels_train,
            #     epochs=200,
            #     verbose=1,
            #     validation_data=(
            #         [components_validation, sizes_validation, num_patches_validation], labels_validation),
            #     callbacks=[tf.keras.callbacks.EarlyStopping(monitor=loss_metric,
            #                                                 mode='max',
            #                                                 patience=n_early_stopping_epochs,
            #                                                 restore_best_weights=True),
            #                                                 tensorboard_callback],
            #     batch_size=batch_size,
            #     class_weight=class_weight
            # )
            fold_dict = folds_training_dicts[i]
            if with_pesto:
                fold_dict['components_train'], fold_dict['components_validation'], fold_dict['components_test'] = utils.filter_with_ablation_string(ablation_string,fold_dict['components_train'],fold_dict['components_validation'],fold_dict['components_test']) 
            model = train_fold(architecture_dict,i,fold_dict,model_log_dir,with_pesto,ablation_string)
            yhat_validation = model.predict([fold_dict['components_validation'], fold_dict['sizes_validation'], fold_dict['num_patches_validation']])
            architecture_validation_predictions.append(yhat_validation)
            architecture_validation_labels.append(fold_dict['labels_validation'].numpy())
            
            yhat_test = model.predict([fold_dict['components_test'], fold_dict['sizes_test'], fold_dict['num_patches_test']])
            architecture_test_predictions.append(yhat_test)
            architecture_test_labels.append(fold_dict['labels_test'].numpy())
            architecture_models.append(model)

        all_architecture_labels = np.concatenate(architecture_validation_labels)
        all_architecture_predictions = np.concatenate(architecture_validation_predictions)
        precision, recall, _ = utils.precision_recall_curve(all_architecture_labels, all_architecture_predictions)
        pr_auc = auc(recall, precision)
        print(f'pr_auc is {pr_auc}')
        architecture_dict['val_metric'] = pr_auc
        grid_results.append(architecture_dict)
        
        if pr_auc > max_pr_auc:
            max_pr_auc = pr_auc
            best_models = architecture_models
            best_architecture = f"architecture:{architecture_dict['n_layers']}_{architecture_dict['m_a']}_{architecture_dict['m_b']}_{architecture_dict['m_c']}"
            best_architecture_test_predictions = architecture_test_predictions
            best_architecture_test_labels = architecture_test_labels
    
    models_architecture_folder = os.path.join(models_folder_path,best_architecture)+'_second'
    os.makedirs(models_architecture_folder,exist_ok=True)
    results_architecture_folder = os.path.join(results_folder_path,best_architecture)+'_second'
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
    with_pesto = True
    # ABLATION STRING[i] = 1 MEANS THAT WE ARE USING THE I'TH FEATURE FROM THIS LIST OF FEATURES:
    # [patch size , scanNet_ubiq , scanNet_protein , pesto_protein , pesto_dna_rna , pesto_ion , pesto_ligand , pesto_lipid , average_plddt]
    ablation_string = '111111111'

    with_pesto_addition = f'_with_pesto' if with_pesto else ''
    training_name = f'{DATE}{with_pesto_addition}'
    data_for_training_folder_path = os.path.join(paths.patch_to_score_data_for_training_path, f'{training_name}')
    training_name += f'_{ablation_string}'
    models_folder_path = os.path.join(paths.patch_to_score_model_path, f'{training_name}')
    results_folder_path = os.path.join(paths.patch_to_score_results_path, f'{training_name}')
    
    train_models(models_folder_path,results_folder_path,data_for_training_folder_path,with_pesto,ablation_string)


