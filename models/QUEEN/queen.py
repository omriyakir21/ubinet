import pickle
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import matplotlib.pyplot as plt
import joblib
import os
import os.path
import torch
import seaborn as sns
from sklearn import metrics
from datetime import date
import torch
import re
import requests
from tqdm.auto import tqdm
import sys
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

    print('probabilities added and saved with entry dicts')
    # # Insert your table here, if you have more than a few fasta files you wish to embed
    # full_tab_for_embed = pd.read_csv("QUEEN/Supplementary_Table_I_embed_tab_with_sets.tsv", sep='\t')
    # full_tab_for_embed.reset_index(inplace=True, drop=True)
    # # comment out the next two lines when using your table, they are here for the demo
    # ###
    # full_tab_for_embed.drop("esm_embeddings", axis=1, inplace=True)
    # full_tab_for_embed = full_tab_for_embed.head(15)
    # ###
    # print(full_tab_for_embed)
    #
    # # Use this cell together with the previous one, if you have more than a few fasta sequences
    # fasta_list = list(zip(full_tab_for_embed["fasta"].index, full_tab_for_embed["fasta"]))
    #
    #
    # def divide_chunks(fasta_list, n):
    #     # looping till length l
    #     for i in range(0, len(fasta_list), n):
    #         yield fasta_list[i:i + n]
    #
    #
    # # if not os.path.exists("/embeds"):
    # #   os.makedirs("/embeds")
    #
    # list_of_chunks = list(divide_chunks(fasta_list, 10))
    # for chunk_num, chunk in enumerate(list_of_chunks):
    #     fname = "QUEEN/embed_pkl_chunk" + str(chunk_num)
    #     if os.path.isfile(fname):
    #         continue
    #
    #     data = chunk
    #     batch_labels, batch_strs, batch_tokens = batch_converter(data)
    #     batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    #
    #     # Extract per-residue representations (on CPU)
    #     with torch.no_grad():
    #         results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    #     token_representations = results["representations"][33]
    #
    #     # Generate per-sequence representations via averaging
    #     # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    #     sequence_representations = []
    #     for i, tokens_len in enumerate(batch_lens):
    #         sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
    #         # print(i)
    #         # print(tokens_len)
    #         # print(batch_lens)
    #
    #     del results
    #     del token_representations
    #     print("saving " + str(chunk_num))
    #     with open(fname, 'wb') as f:
    #         pickle.dump(sequence_representations, f)
    #
    # # When you do have a previous table that was used to generate the embeddings, we open all the tensors and add them to the table
    # from natsort import humansorted
    #
    # esm_embeds_dir = paths.QUEEN_model_path
    # dir_list = humansorted(os.listdir(esm_embeds_dir))
    # pickle_list = [x for x in pickle_list if "chunk" in x]
    # tensor_list = []
    # for file in pickle_list:
    #     full_path = os.path.join(paths.QUEEN_model_path, file)
    #     with open(full_path, "rb") as f:
    #         current = pickle.load(f)
    #         #        print(current)
    #         tensor_list.extend(current)
    # #        print(tensor_list)
    # with open(os.path.join(paths.QUEEN_model_path, "tensor_list.pkl"), "wb") as f:
    #     pickle.dump(tensor_list, f)
    #
    # np_list = []
    # for i, ten in enumerate(tensor_list):
    #     ten = ten.detach().numpy()
    #     np_list.append(ten)
    #
    # full_tab_for_embed["esm_embeddings"] = pd.Series(np_list)
    #
    # with open(os.path.join(paths.QUEEN_model_path, "tab_for_pred.pkl"), "wb") as f:
    #     pickle.dump(full_tab_for_embed, f)
    #
    # # load model
    # model_location = os.path.join(paths.QUEEN_model_path, "QUEEN_MLPmodel_final.pkl")
    # with open(model_location, "rb") as f:
    #     QUEEN_model = pickle.load(f)
    #
    # y_test = QUEEN_model.predict(full_tab_for_embed["esm_embeddings"].to_list())
    # inv_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 10, 9: 12, 10: 14, 11: 24}
    # y_test_transformed = np.array([inv_map[x] for x in y_test])
    # full_tab_for_embed["y_pred"] = pd.Series(y_test_transformed)
    # print("These are the predicted labels:")
    # print(y_test_transformed)
    # print("this is the final table")
    # print(full_tab_for_embed)
