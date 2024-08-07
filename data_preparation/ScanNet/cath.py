import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import cath_utils
import paths

if __name__ == "__main__":
    cath_df = cath_utils.make_cath_df_new(os.path.join(paths.cath_path, "cath_b.20230204.txt"))
    names_list, sizes_list, sequence_list, full_names_list, pdb_names_with_chains_lists = cath_utils.list_creation(
        "propagatedFullPssmFile")
    structuresDicts = cath_utils.create_dictionaries(names_list, sizes_list, sequence_list, full_names_list,
                                                     pdb_names_with_chains_lists)
    inCath, notInCath, cnt = cath_utils.count_in_cath(cath_df, structuresDicts)

    print(len(structuresDicts))
    print(cnt)
    print(inCath)
    print(notInCath)
    # print(cath_df)
    cath_utils.find_chains_in_cath(cath_df, structuresDicts)
    cath_utils.add_classifications_for_dict(cath_df, structuresDicts, 4)
    matHomologous = cath_utils.neighbor_mat_new(structuresDicts)
    graphHomologous = cath_utils.csr_matrix(matHomologous)
    homologous_components, homologousLabels = cath_utils.connected_components(csgraph=graphHomologous, directed=False,
                                                                              return_labels=True)
    print(names_list)
    print(sizes_list)
    print(sum(sizes_list))
    print(homologous_components)
    print(homologousLabels)
    print("Done2")
    relatedChainsLists = cath_utils.create_related_chainslist(homologous_components, homologousLabels)
    print("Done3")
    clusterSizes = cath_utils.create_cluster_sizes_list(relatedChainsLists, sizes_list)
    print("Done4")
    sublists, sublistsSum = cath_utils.divide_clusters(clusterSizes)
    print("Done5")

    print(relatedChainsLists)
    print(clusterSizes)
    print(sublists)
    print(sublistsSum)

    chainLists = cath_utils.sublists_to_chain_lists(sublists, relatedChainsLists, full_names_list)
    chainDict = cath_utils.chain_lists_to_chain_index_dict(chainLists)
    print(chainLists)
    print(chainDict)
    full_pssm_file_path = os.path.join(os.path.join(paths.PSSM_path, 'propagatedPssmWithAsaFile.txt'))
    cath_utils.divide_pssm(chainDict, full_pssm_file_path)
