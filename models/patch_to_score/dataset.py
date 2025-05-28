import sys
import os
from typing import Union
import tensorflow as tf
from data_preparation.ScanNet.db_creation_scanNet_utils import load_as_pickle


class PatchToScoreDataset:
    def __init__(self, fold: dict, use_pesto: bool, ablation_string: str, use_coordinates: bool):
        self.ablation_string = ablation_string
        self.use_pesto = use_pesto
        self.use_coordinates = use_coordinates
        self.fold = fold

        self.train_set = self.fold['components_train'], 
        self.validation_set = self.fold['components_validation'], 
        self.test_set = self.fold['components_test']

        self.train_labels = self.fold['labels_train']
        self.validation_labels = self.fold['labels_validation']
        self.test_labels = self.fold['labels_test']

        self.train_sizes = self.fold['sizes_train']
        self.validation_sizes = self.fold['sizes_validation']
        self.test_sizes = self.fold['sizes_test']

        self.train_num_patch = self.fold['num_patches_train']
        self.validation_num_patch = self.fold['num_patches_validation']
        self.test_num_patch = self.fold['num_patches_test']
            
        if self.use_pesto:
            self.train_set, self.validation_set, self.test_set = self.filter_with_ablation_string(self.ablation_string,
                                                                                                  self.fold['components_train'],
                                                                                                  self.fold['components_validation'], 
                                                                                                  self.fold['components_test'])
        # TODO: this is the current solution, instead of bootstrapping datasets 
        self.train_set = [self.train_set]
        self.validation_set = [self.validation_set]
        self.test_set = [self.test_set]
        
        if self.use_coordinates:
            self.train_set.append(self.fold['coordinates_train'])
            self.validation_set.append(self.fold['coordinates_validation'])
            self.test_set.append(self.fold['coordinates_test'])

    @staticmethod
    def filter_with_ablation_string(ablation_string,components_train,components_validation,components_test):
        mask = tf.constant([bool(int(x)) for x in ablation_string], dtype=tf.bool)
        train_filtered_components = tf.boolean_mask(components_train, mask, axis=2)
        validation_filtered_components = tf.boolean_mask(components_validation, mask, axis=2)
        test_filtered_components = tf.boolean_mask(components_test, mask, axis=2)
        return train_filtered_components,validation_filtered_components,test_filtered_components


class PatchToScoreCrossValidationDataset:
    def __init__(self, path: str, use_pesto: bool, ablation_string: str, max_folds: Union[None, int], use_coordinates: bool = False):
        self.path = path
        self.ablation_string = ablation_string
        self.use_pesto = use_pesto
        self.use_coordinates = use_coordinates
        self.max_folds = max_folds
        self.fold_dicts = load_as_pickle(path)
        self.fold_datasets = []

        for i, fold in enumerate(self.fold_dicts):
            if (self.max_folds is not None) and ((i + 1) > self.max_folds):
                print('Got to max folds - cutting cross validation dataset')
                break
            fold_dataset = PatchToScoreDataset(fold, self.use_pesto, self.ablation_string, self.use_coordinates)
            self.fold_datasets.append(fold_dataset)

        print('folds amount:', len(self.fold_datasets))