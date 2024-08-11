import pickle
import subprocess
import sys
from plistlib import load

import pandas as pd
import numpy as np
import os
from Bio.PDB import MMCIFParser
from data_development_utils import Protein



# allProteinsDict = loadPickle('/home/iscb/wolfson/omriyakir/UBDModel/aggregateFunctionMLP/allProteinInfo.pkl')

# saveAsPickle(sequences,r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\aggregateFunctionMLP\dataForTraining23_3\allProteinSequences')

# cluster_indices = loadPickle(
#     r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\aggregateFunctionMLP\dataForTraining23_3\clusterIndices.pkl')
# clustersParticipantsList = loadPickle(
#     r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\aggregateFunctionMLP\dataForTraining23_3\clustersParticipantsList.pkl')
# representative_indices = loadPickle(
#     r'C:\Users\omriy\UBDAndScanNet\newUBD\UBDModel\aggregateFunctionMLP\dataForTraining23_3\representative_indices.pkl')




# saveAsPickle(cluster_indices, '/home/iscb/wolfson/omriyakir/UBDModel/aggregateFunctionMLP/dataForTraining23_3/' + 'clusterIndices')
# saveAsPickle(representative_indices, '/home/iscb/wolfson/omriyakir/UBDModel/aggregateFunctionMLP/dataForTraining23_3/' + 'representative_indices')
# saveAsPickle(clustersParticipantsList, '/home/iscb/wolfson/omriyakir/UBDModel/aggregateFunctionMLP/dataForTraining23_3/' + 'clustersParticipantsList')

