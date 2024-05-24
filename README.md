# DEMO

This repository includes codes to replicate the results in paper

CLAIRE: A Contrastive Learning-based Predictor for EC number of chemical reactions

as well as demonstrates how to use CLAIRE to predict EC number for chemical reactions. 

**1.Environment setup**

In terminal
```
conda activate rxnfp
cd CLAIRE/
```

In Python, import the relevant packages
```
from dev.prediction.inference_EC import infer_maxsep
import pickle
import numpy as np
import pandas as pd
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)
```

**2.Make predictions for a list of reactions in SMILES format**

***rxnfp Embedding：***

Note that multiple reactants and products are seaparated by "."; reactants and products are separated by ">>".


```python
example_rxns = ["NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.NCCC=O.O>>NCCC(=O)O", "C=C(C)CCOP(=O)([O-])OP(=O)([O-])[O-].CC(C)=CCOP(=O)(O)OP(=O)(O)O>>CC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCCC(C)=CCOP(=O)(O)OP(=O)(O)O", "N.NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.O=C([O-])CCC(=O)C(=O)[O-].[H+]>>N[C@@H](CCC(=O)[O-])C(=O)[O-]"]
rxnfp = rxnfp_generator.convert_batch(example_rxns)
```

***drfp embedding***

save the reaction SMILES in "my_rxn_smiles.txt", then run the following to save drfp embeddings in "my_rxn_fps.pkl"
```
drfp my_rxn_smiles.txt my_rxn_fps.pkl -d 256
```
where -d is the dimension of the embeddings

***concat rxnfp and drfp***

```python
test_data = []
for ind, item in enumerate(rxnfp):
    rxn_emb = np.concatenate((np.reshape(item, (1,256)), np.reshape(drfp[ind], (1,256))), axis=1)
    test_data.append(rxn_emb)
test_data = np.concatenate(test_data,axis=0)
```

**5.Inference：**
```python
# EC calling results using maximum separation
infer_maxsep(train_data, test_data, train_labels, test_tags, test_labels, pretrained_model,out_filename='./dev/results/demo', gmm = './dev/GMM/gmm_ensumble.pkl')
```
