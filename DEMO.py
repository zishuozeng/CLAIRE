"""
    inference
"""
from dev.prediction.inference_EC import infer_maxsep
import pickle
import numpy as np
import pandas as pd
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)


### train data
# train_embedding 512-dim
train_file = './dev/data/model_lookup_train.pkl'
with open (train_file, 'rb') as file:
    train_data = pickle.load(file)

# train_labels
labels_file = './dev/data/pred_rxn_EC123/labels_train_ec3.pkl'
with open (labels_file, 'rb') as file:
    train_labels = pickle.load(file)


### test data
### use our test data
# test_embedding 256-dim
test_file = './dev/data/model_lookup_test.pkl'
with open (test_file, 'rb') as file:
    test_data = pickle.load(file)

# test_labels
labels_file = './dev/data/pred_rxn_EC123/labels_test_ec3.pkl'
with open (labels_file, 'rb') as file:
    test_labels = pickle.load(file)


test_data = np.r_[test_data[:100], test_data[-50:]]
test_labels = test_labels[:100] + test_labels[-50:]


test_tags = []
for i in range(len(test_data)):
    test_tags.append('rxn_' + str(i))
print(len(test_tags))

# 训练好的模型
pretrained_model = './dev/results/model/pred_rxn_EC123/layer5_node1280_triplet2000_final.pth'

# report_metrics = True : Evaluate the predictions
# EC calling results using maximum separation
infer_maxsep(train_data, test_data, train_labels, test_tags, test_labels, pretrained_model,out_filename='./dev/results/demo', gmm = './dev/GMM/gmm_ensumble.pkl')

# 预测结果保存至 results/test_prediction.csv
