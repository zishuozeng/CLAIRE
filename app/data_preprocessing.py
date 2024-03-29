import pickle
import pandas as pd
import csv
from tqdm import tqdm
import numpy as np
import torch

###############################################################################################################################
###############################################################################################################################
######  数据集：只取EC的前三位作为标签  
#################################################

### train
rxn_file = './data/inputs/pred_rxn_EC3/rxn_ec_emb.pkl'
with open(rxn_file,'rb') as f:
    rxns = pickle.load(f)

samples_file = './data/inputs/pred_rxn_EC3/train_augmented.csv'
with open(samples_file, 'r', encoding='utf-8') as f:
    samples = csv.reader(f)
    next(samples)
    train = list(samples)

model_lookup_train = []
labels_train = []
for ind, item in tqdm(enumerate(train),total=len(train)):   
    # 检索 rxn 的 Emb 
    if item[0] in rxns.keys():
        rxn = rxns[item[0]] 
        model_lookup_train.append(rxn)
        labels_train.append(item[1])
    else:
        # print(f'找不到rxn{ind}:{samples[ind]}')
        continue

print(len(model_lookup_train))  # 166918


output_file = './data/inputs/model_lookup_train.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(np.array(model_lookup_train), file)

output_file = './data/inputs/pred_rxn_EC3/labels_train.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(labels_train, file)



### test
rxn_file = './data/inputs/pred_rxn_EC3/rxn_ec_emb.pkl'
with open(rxn_file,'rb') as f:
    rxns = pickle.load(f)

samples_file = './data/inputs/pred_rxn_EC3/test_augmented.csv'
with open(samples_file, 'r', encoding='utf-8') as f:
    samples = csv.reader(f)
    next(samples)
    test = list(samples)

model_lookup_test = []
labels_test = []
for ind, item in tqdm(enumerate(test),total=len(test)):   
    # 检索 rxn 的 Emb 
    if item[0] in rxns.keys():
        rxn = rxns[item[0]] 
        model_lookup_test.append(rxn)
        labels_test.append(item[1])
    else:
        # print(f'找不到rxn{ind}:{test[ind]}')
        continue

print(len(model_lookup_test))    # 18816

output_file = './data/inputs/model_lookup_test.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(np.array(model_lookup_test), file)

output_file = './data/inputs/pred_rxn_EC3/labels_test.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(labels_test, file)


#############################################
#####  数据集：只取EC的前三位作为标签  制作 esm_emb [tensor格式] [字典格式]   
#############################################
### train
rxn_file = './data/inputs/pred_rxn_EC3/rxn_ec_emb.pkl'
with open(rxn_file,'rb') as f:
    rxns = pickle.load(f)

samples_file = './data/inputs/pred_rxn_EC3/train_augmented.csv'
with open(samples_file, 'r', encoding='utf-8') as f:
    samples = csv.reader(f)
    next(samples)
    samples = list(samples)

model_lookup_train = {}
for ind, item in tqdm(enumerate(samples),total=len(samples)):    
    # 检索 rxn 的 Emb 
    if item[0] in rxns.keys():
        rxn = rxns[item[0]] 
        rxn = torch.Tensor(rxn)
    else:
        # print(f'找不到rxn{ind}:{samples[ind]}')
        continue

    
    
    if item[1] in model_lookup_train.keys():
        model_lookup_train[item[1]].append(rxn)
    else:
        temp_list = []
        temp_list.append(rxn)
        
        model_lookup_train[item[1]] = temp_list


print(len(model_lookup_train))  # unique_ec: 176


output_file = './data/inputs/pred_rxn_EC3/esm_emb_dict.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(model_lookup_train, file)



##############################################################################################################################
##############################################################################################################################
#####  数据集：只取EC的前一位作为标签 
################################################

### train
rxn_file = './data/inputs/pred_rxn_EC1/rxn_ec_emb.pkl'
with open(rxn_file,'rb') as f:
    rxns = pickle.load(f)

samples_file = './data/inputs/pred_rxn_EC1/train_augmented.csv'
with open(samples_file, 'r', encoding='utf-8') as f:
    samples = csv.reader(f)
    next(samples)
    train = list(samples)

model_lookup_train = []
labels_train = []
for ind, item in tqdm(enumerate(train),total=len(train)):   
    # 检索 rxn 的 Emb 
    if item[0] in rxns.keys():
        rxn = rxns[item[0]] 
        model_lookup_train.append(rxn)
        label = item[1].split('.')[0]
        labels_train.append(label)
    else:
        # print(f'找不到rxn{ind}:{samples[ind]}')
        continue

print(len(model_lookup_train))  # 166918


output_file = './data/inputs/model_lookup_train.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(np.array(model_lookup_train), file)

output_file = './data/inputs/pred_rxn_EC1/labels_train.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(labels_train, file)



## test
rxn_file = './data/inputs/pred_rxn_EC1/rxn_ec_emb.pkl'
with open(rxn_file,'rb') as f:
    rxns = pickle.load(f)

samples_file = './data/inputs/pred_rxn_EC1/test_augmented.csv'
with open(samples_file, 'r', encoding='utf-8') as f:
    samples = csv.reader(f)
    next(samples)
    test = list(samples)

model_lookup_test = []
labels_test = []
for ind, item in tqdm(enumerate(test),total=len(test)):   
    # 检索 rxn 的 Emb 
    if item[0] in rxns.keys():
        rxn = rxns[item[0]] 
        model_lookup_test.append(rxn)
        label = item[1].split('.')[0]
        labels_test.append(label)
    else:
        # print(f'找不到rxn{ind}:{test[ind]}')
        continue

print(len(model_lookup_test))    # 18816

output_file = './data/inputs/model_lookup_test.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(np.array(model_lookup_test), file)

output_file = './data/inputs/pred_rxn_EC1/labels_test.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(labels_test, file)


#############################################
#####  数据集：只取EC的第一位作为标签  制作 esm_emb [tensor格式] [字典格式]   
#############################################
### train
rxn_file = './data/inputs/pred_rxn_EC1/rxn_ec_emb.pkl'
with open(rxn_file,'rb') as f:
    rxns = pickle.load(f)

samples_file = './data/inputs/pred_rxn_EC1/train_augmented.csv'
with open(samples_file, 'r', encoding='utf-8') as f:
    samples = csv.reader(f)
    next(samples)
    samples = list(samples)

model_lookup_train = {}
for ind, item in tqdm(enumerate(samples),total=len(samples)):    
    # 检索 rxn 的 Emb 
    if item[0] in rxns.keys():
        rxn = rxns[item[0]] 
        rxn = torch.Tensor(rxn)
    else:
        # print(f'找不到rxn{ind}:{samples[ind]}')
        continue

    label = item[1].split('.')[0]

    if label in model_lookup_train.keys():
        model_lookup_train[label].append(rxn)
    else:
        temp_list = []
        temp_list.append(rxn)
        
        model_lookup_train[label] = temp_list


print(len(model_lookup_train))  # unique_ec: 8


output_file = './data/inputs/pred_rxn_EC1/esm_emb_dict.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(model_lookup_train, file)


##############################################################################################################################
##############################################################################################################################
#####  数据集：只取EC的前两位作为标签 
################################################

### train
rxn_file = './data/inputs/pred_rxn_EC2/rxn_ec_emb.pkl'
with open(rxn_file,'rb') as f:
    rxns = pickle.load(f)

samples_file = './data/inputs/pred_rxn_EC2/train_augmented.csv'
with open(samples_file, 'r', encoding='utf-8') as f:
    samples = csv.reader(f)
    next(samples)
    train = list(samples)

model_lookup_train = []
labels_train = []
for ind, item in tqdm(enumerate(train),total=len(train)):   
    # 检索 rxn 的 Emb 
    if item[0] in rxns.keys():
        rxn = rxns[item[0]] 
        model_lookup_train.append(rxn)
        label_list = item[1].split('.')[:2]
        label = '.'.join(label_list)
        labels_train.append(label)
    else:
        # print(f'找不到rxn{ind}:{samples[ind]}')
        continue

print(len(model_lookup_train))  # 166918


output_file = './data/inputs/model_lookup_train.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(np.array(model_lookup_train), file)

output_file = './data/inputs/pred_rxn_EC2/labels_train.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(labels_train, file)



### test
rxn_file = './data/inputs/pred_rxn_EC2/rxn_ec_emb.pkl'
with open(rxn_file,'rb') as f:
    rxns = pickle.load(f)

samples_file = './data/inputs/pred_rxn_EC2/test_augmented.csv'
with open(samples_file, 'r', encoding='utf-8') as f:
    samples = csv.reader(f)
    next(samples)
    test = list(samples)

model_lookup_test = []
labels_test = []
for ind, item in tqdm(enumerate(test),total=len(test)):   
    # 检索 rxn 的 Emb 
    if item[0] in rxns.keys():
        rxn = rxns[item[0]] 
        model_lookup_test.append(rxn)
        label_list = item[1].split('.')[:2]
        label = '.'.join(label_list)
        labels_test.append(label)
    else:
        # print(f'找不到rxn{ind}:{test[ind]}')
        continue

print(len(model_lookup_test))    # 18816

output_file = './data/inputs/model_lookup_test.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(np.array(model_lookup_test), file)

output_file = './data/inputs/pred_rxn_EC2/labels_test.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(labels_test, file)


#############################################
#####  数据集：只取EC的前两位作为标签  制作 esm_emb [tensor格式] [字典格式]   
#############################################
### train
rxn_file = './data/inputs/pred_rxn_EC2/rxn_ec_emb.pkl'
with open(rxn_file,'rb') as f:
    rxns = pickle.load(f)

samples_file = './data/inputs/pred_rxn_EC2/train_augmented.csv'
with open(samples_file, 'r', encoding='utf-8') as f:
    samples = csv.reader(f)
    next(samples)
    samples = list(samples)

model_lookup_train = {}
for ind, item in tqdm(enumerate(samples),total=len(samples)):    
    # 检索 rxn 的 Emb 
    if item[0] in rxns.keys():
        rxn = rxns[item[0]] 
        rxn = torch.Tensor(rxn)
    else:
        # print(f'找不到rxn{ind}:{samples[ind]}')
        continue

    
    label_list = item[1].split('.')[:2]
    label = '.'.join(label_list)

    if label in model_lookup_train.keys():
        model_lookup_train[label].append(rxn)
    else:
        temp_list = []
        temp_list.append(rxn)
        
        model_lookup_train[label] = temp_list


print(len(model_lookup_train))  # unique_ec: 63


output_file = './data/inputs/pred_rxn_EC2/esm_emb_dict.pkl'
with open(output_file, 'wb') as file:
    pickle.dump(model_lookup_train, file)

