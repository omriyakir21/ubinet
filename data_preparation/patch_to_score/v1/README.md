## Patch2Score Dataset - v1

**Bootstrap name** - `v1` <br>
Meaning, when creating a configuration file, and bootstrapping the dataset, this dataset will be named `v1` <br>

**Background** - This is the second version of the Patch2Score dataset.
The first version is named 'v0', and it's source code can be found under `data_preparation/patch_to_score`. <br>

**Key differences from `v0`** :
1. Contains geometrical data (x,y,z per amino-acid)
2. Data is not aggregated: predictions are saved per amino-acid, instead of average across patch.<br>
The idea is that the aggregation step will happen in the model, which can now be more complex then average, and most importatnly - learned. 
3. Refactor: code structure refactor, save format, etc.

**Pipeline** :
1. `compute_global_values` : computes the 90th percentile of scannet ubiquitin binding score 
2. `create_protein_chains` : create unscaled `PatchToScoreProteinChain` objects
3. `scale` : scales the `PatchToScoreProteinChain` objects
4. `partition` : partition the chains to folds
