"""
    inference
"""
from inference_EC import infer_maxsep
import pickle
import numpy as np
import pandas as pd
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)


### train data
# train_embedding 256-dim
train_file = './data/inputs/pred_rxn_EC12/model_lookup_train.pkl'
with open (train_file, 'rb') as file:
    train_data = pickle.load(file)

# train_labels
labels_file = './data/inputs/pred_rxn_EC12/labels_train.pkl'
with open (labels_file, 'rb') as file:
    train_labels = pickle.load(file)


### test data
'''
### use our test data
# test_embedding 256-dim
test_file = './data/inputs/pred_rxn_EC12/model_lookup_test.pkl'
with open (test_file, 'rb') as file:
    test_data = pickle.load(file)

# test_labels
labels_file = './data/inputs/pred_rxn_EC12/labels_test.pkl'
with open (labels_file, 'rb') as file:
    test_labels = pickle.load(file)

test_data = np.r_[test_data[:100], test_data[-50:]]
test_labels = test_labels[:100] + test_labels[-50:]
'''
### example

model, tokenizer = get_default_model_and_tokenizer()
rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)

example_rxn = ["C/C(C=O)=C\CC/C(C)=C/C=O.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.[H+].O>>COC(=O)CCCCCCCC=O", "CC(C=O)c1ccccc1.O.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1>>CC(C(=O)O)c1ccccc1"]

fps = rxnfp_generator.convert_batch(example_rxn)
test_data = np.array(fps)
test_labels = ['1.2', '1.2']
###

test_tags = []
for i in range(len(test_data)):
    test_tags.append('rxn_' + str(i))
print(len(test_tags))

# 训练好的模型
pretrained_model = './data/model/pred_rxn_EC12/random_10-4_triplet2000_final.pth'

# report_metrics = True : Evaluate the predictions
# EC calling results using maximum separation
infer_maxsep(train_data, test_data, train_labels, test_tags, test_labels, pretrained_model, report_metrics=True, pretrained=False, gmm = './data/pretrained/gmm_ensumble.pkl')

# 预测结果保存至 results/inputs/test_maxsep.csv
