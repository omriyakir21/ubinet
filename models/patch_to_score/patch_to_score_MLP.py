import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np    
from sklearn.metrics import auc
import paths
import patch_to_score_MLP_utils as utils
import tensorflow as tf
from data_preparation.ScanNet.db_creation_scanNet_utils import load_as_pickle,save_as_pickle
from results.patch_to_score.patch_to_score_result_analysis import get_best_architecture_path

def train_models(directory_name):
    folds_training_dicts = load_as_pickle(os.path.join(paths.patch_to_score_data_for_training_path,
                                                                        'folds_traning_dicts.pkl'))
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)

    total_aucs = []
    n_layers = int(sys.argv[1])
    m_a = int(sys.argv[2])
    m_b_values = [128, 256, 512]
    m_c = int(sys.argv[3])
    batch_size = 1024
    n_early_stopping_epochs = 12
    for m_b in m_b_values:
        architecture_dir_path = os.path.join(directory_name,f'architecture:{n_layers}_{m_a}_{m_b}_{m_c}')
        if not os.path.exists(architecture_dir_path):
            os.mkdir(architecture_dir_path)
        all_predictions = []
        all_labels = []
        architecture_aucs = []
        for i in range(len(folds_training_dicts)):
            tf.keras.backend.clear_session()
            if 'model' in locals():
                del model
            model = utils.build_model_concat_size_and_n_patches_same_number_of_layers(m_a, m_b, m_c, n_layers)
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
                                                            restore_best_weights=True)],
                batch_size=batch_size,
                class_weight=class_weight

            )

            yhat_validation = model.predict(
                [components_validation, sizes_validation, num_patches_validation])
            precision, recall, _ = utils.precision_recall_curve(labels_validation, yhat_validation)
            pr_auc = auc(recall, precision)
            architecture_aucs.append(((m_a, m_b, m_c, n_layers,
                                    n_early_stopping_epochs, batch_size, i), pr_auc))
            all_predictions.append(yhat_validation)
            all_labels.append(labels_validation.numpy())
            model.save(os.path.join(architecture_dir_path, 'model'+str(i)+'.keras'))

        all_labels = np.concatenate(all_labels)
        all_predictions = np.concatenate(all_predictions)
        precision, recall, _ = utils.precision_recall_curve(all_labels, all_predictions)
        pr_auc = auc(recall, precision)
        total_aucs.append(((m_a, m_b, m_c, n_layers,
                        n_early_stopping_epochs, batch_size), pr_auc))
        np.save(os.path.join(architecture_dir_path,f'predictions.npy'),all_predictions)
        np.save(os.path.join(architecture_dir_path,f'labels.npy'),all_labels)
        save_as_pickle(architecture_aucs,os.path.join(architecture_dir_path,f'architecture_aucs.pkl'))
    save_as_pickle(total_aucs, os.path.join(directory_name, f'totalAucs:n_layers:{str(n_layers)}_m_a:{str(m_a)}_m_c:{str(m_c)}.pkl'))

def predict_over_test_set(model_dir_path):
    models_dir_path = get_best_architecture_path(models_dir_path)
    models = []
    for filename in os.listdir(models_dir_path):
        if filename.endswith('.keras'):
            models.append(tf.keras.models.load_model(os.path.join(models_dir_path, filename)))
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
        predictions = []
        for model in models:
            predictions.append(model.predict([components_test, sizes_test, num_patches_test]))
        predictions = np.mean(predictions, axis=0)
        all_predictions.append(predictions)
        all_labels.append(labels_test.numpy())
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)
    np.save(os.path.join(models_dir_path,'predictions_test.npy'),all_predictions)
    np.save(os.path.join(models_dir_path,'labels_test.npy'),all_labels)

    


if __name__ == "__main__":
    train_models(paths.with_MSA_50_plddt_0304_dir)
    # predict_over_test_set(paths.with_MSA_50_plddt_0304_dir)
    # print('done')
