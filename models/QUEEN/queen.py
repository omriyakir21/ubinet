import pickle
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
# import matplotlib.pyplot as plt
# import joblib
# import os
import os.path
# import torch
# import seaborn as sns
# from sklearn import metrics
# from datetime import date
import torch
# import re
# import requests
# from tqdm.auto import tqdm
# import sys
import esm
import paths

from transformers import EsmTokenizer, EsmForSequenceClassification
from data_preparation.ScanNet.db_creation_scanNet_utils import save_as_pickle, load_as_pickle


# ! git clone https://github.com/Furman-Lab/QUEEN/

def split_list(original_list, num_sublists):
    sublist_size = len(original_list) // num_sublists
    remainder = len(original_list) % num_sublists

    result = []
    index = 0

    for i in range(num_sublists):
        sublist_length = sublist_size + 1 if i < remainder else sublist_size
        sublist = original_list[index:index + sublist_length]
        result.append(sublist)
        index += sublist_length

    return result


if __name__ == "__main__":

    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmForSequenceClassification.from_pretrained("facebook/esm2_t6_8M_UR50D",
                                                         problem_type="multi_label_classification")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.cuda.empty_cache()
    model = model.to(device)
    model = model.eval()

    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    entryDicts = load_as_pickle(os.path.join(paths.entry_dicts_path, 'list_of_entry_dicts.pkl'))
    data = [(entry['entry'], entry['reference_sequence']) for entry in entryDicts]
    batch_size = 100
    # Split data into batches
    batches = split_list(data, len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0))

    all_probabilities = []
    for data in batches:
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]

        # Extract per-residue representations (on CPU)
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            # average on the protein length, to obtain a single vector per fasta
            sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

        # If not previous tab, we generate it here:
        full_tab_for_embed = pd.DataFrame()
        np_list = []
        # Detach the tensors to obtain a numpy array
        for i, ten in enumerate(sequence_representations):
            ten = ten.detach().numpy()
            np_list.append(ten)
        full_tab_for_embed["esm_embeddings"] = pd.Series(np_list)

        # load model
        model_location = os.path.join(paths.QUEEN_model_path, 'QUEEN_MLPmodel_final.pkl')
        with open(model_location, "rb") as f:
            QUEEN_model = pickle.load(f)

        y_prob = QUEEN_model.predict_proba(full_tab_for_embed["esm_embeddings"].to_list())
        all_probabilities.extend(y_prob)

    for entry, prob in zip(entryDicts, all_probabilities):
        entry['probabilities'] = prob

    save_as_pickle(entryDicts, os.path.join(paths.entry_dicts_path, 'list_of_entry_dicts_with_probabilities.pkl'))

