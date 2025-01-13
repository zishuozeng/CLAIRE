
This repository includes codes to run the model in paper

_CLAIRE: A Contrastive Learning-based Predictor for EC number of chemical reactions_

to predict EC numbers for chemical reactions. 

# 1.Environment setup

In terminal
```
cd CLAIRE/
conda create -n claire python==3.10
conda activate claire
pip install -r requirements.txt
```
Install `torch`：You may install GPU or CPU version of `torch`.

```
conda install pytorch==1.11.0 cpuonly -c pytorch (CPU)
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch (GPU)
```

Run the following to install rxnfp:
```
bash rxnfp_env.sh
```

# 2.Data
You can download the data ([*inputs*](https://zenodo.org/records/14635841)) and place it in the `CLAIRE/dev/` directory, which contains 'train_augmented.csv', 'test_augmented.csv', 'model_lookup' and 'embedding'.
```
-train_augmented.csv and test_augmented.csv are the raw data, and model_lookup_train.pkl and model_lookup_test.pkl are the embedding of rxn.
-The embedding folder is a detailed procedure for embedding the rxn.
-pred_rxn_ECxxx is a three-level division of ec, consisting of labels and esm_emb :{'EC1':[tensor1, tensor2, ... ,], 'EC2':[tensor1, tensor2, ... ... .....}.esm_emb is for training purposes only
```

# 3.How to use

**(1). Run DRFP embeddings**

Suppose you have three query reactions to be predicted (shown below), saved in a txt file ("my_rxn_smiles.txt"). 
Note that multiple reactants and products are seaparated by "."; reactants and products are separated by ">>".

```txt
NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.NCCC=O.O>>NCCC(=O)O
C=C(C)CCOP(=O)([O-])OP(=O)([O-])[O-].CC(C)=CCOP(=O)(O)OP(=O)(O)O>>CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP(=O)(O)OP(=O)(O)O
N.NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.O=C([O-])CCC(=O)C(=O)[O-].[H+]>>N[C@@H](CCC(=O)[O-])C(=O)[O-]
```

Activate the `claire` environment:
```
cd CLAIRE/
conda activate claire
```

Run the following command to obtain DRFP embeddings and save it in "my_rxn_fps.pkl"
```
drfp my_rxn_smiles.txt my_rxn_fps.pkl -d 256
```
where -d is the dimension of the embeddings


**(2). Run rxnfp embeddings**

Activate the rxnfp environment:

In Python, import the relevant packages
```python
from dev.prediction.inference_EC import inference
import pickle
import numpy as np
import pandas as pd
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)
```

compute for the rxnfp embeddings
```python
model, tokenizer = get_default_model_and_tokenizer()
rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
example_rxns = ["NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.NCCC=O.O>>NCCC(=O)O", "C=C(C)CCOP(=O)([O-])OP(=O)([O-])[O-].CC(C)=CCOP(=O)(O)OP(=O)(O)O>>CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP(=O)(O)OP(=O)(O)O", "N.NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.O=C([O-])CCC(=O)C(=O)[O-].[H+]>>N[C@@H](CCC(=O)[O-])C(=O)[O-]"]
rxnfp = rxnfp_generator.convert_batch(example_rxns)
pickle.dump(rxnfp, open('rxnfp_emb.pkl', 'wb'))
```

**(3). Concatenate the rxnfp and drfp embeddings**

```python
drfp = pickle.load(open('my_rxn_fps.pkl', 'rb'))
rxnfp = pickle.load(open('rxnfp_emb.pkl', 'rb'))
test_data = []

for ind, item in enumerate(rxnfp):
    rxn_emb = np.concatenate((np.reshape(item, (1,256)), np.reshape(drfp[ind], (1,256))), axis=1)
    test_data.append(rxn_emb)

test_data = np.concatenate(test_data,axis=0)
```
**(4). Make predictions on the concatenated embeddings**

Activate the claire environment:
```python
train_data = pickle.load(open ('data/model_lookup_train.pkl', 'rb'))
train_labels = pickle.load(open ('data/pred_rxn_EC123/labels_train_ec3.pkl', 'rb'))
# input your test_labels
test_labels = None
test_tags = ['rxn_' + str(i) for i in range(len(test_data))]

# EC calling results using maximum separation
pretrained_model = '../results/model/pred_rxn_EC123/layer5_node1280_triplet2000_final.pth'
inference(train_data, test_data, train_labels, test_tags,test_labels, pretrained_model, evaluation=True, topk=3, gmm = '../gmm/gmm_ensumble.pkl')
```
The prediction results are saved in `dev/test_prediction.csv`.

This project uses part of codes (the gmm functions) from the [*CLEAN*](https://github.com/tttianhao/CLEAN/) software developed by the Department of Chemical and Biomolecular Engineering at the University of Illinois Urbana-Champaign.
