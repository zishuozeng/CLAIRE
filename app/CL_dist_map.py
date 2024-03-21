import pickle
import pandas as pd
import csv
from tqdm import tqdm
import numpy as np
import torch

#############################################
##### 1 ###### 制作 model_lookup 及 labels
#############################################
# seq_file = '/root/CLEAN/app/data/seq_embeddings.pkl'
# with open(seq_file,'rb') as f:
#     seqs = pickle.load(f)

# rxn_file = '/root/CLEAN/app/data/rxn_embeddings_dict.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)

# samples_file = '/root/CLEAN/app/data/samples.csv'
# with open(samples_file, 'r', encoding='utf-8') as f:
#     samples = csv.reader(f)
#     next(samples)
#     samples = list(samples)

# model_lookup = []
# labels = []
# for ind, item in tqdm(enumerate(samples),total=len(samples)):    
#     # 检索 rxn 的 Emb 
#     if item[0] in rxns.keys():
#         rxn = rxns[item[0]]
#     else:
#         # print(f'找不到rxn{ind}:{samples[ind]}')
#         continue

#     # 检索 seq 的 Emb
#     if item[3] in seqs.keys():
#         seq = seqs[item[3]]
#     else:
#         # print(f'找不到seq{ind}:{samples[ind]}')
#         continue
#     model_lookup.append(list(rxn)+list(seq[0]))    

#     # 统计 labels
#     labels.append(item[2])

# output_file = '/root/CLEAN/app/data/model_lookup.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(np.array(model_lookup), file)

# output_file = '/root/CLEAN/app/data/labels.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(labels, file)
    

#############################################
########### 制作 model_dist
#############################################        
# import numpy as np
# import torch
# import pickle

# def dist_map_helper(keys1, lookup1, keys2, lookup2):
#     dist = {}
#     for i, key1 in tqdm(enumerate(keys1), total=len(keys1)):
#         current = lookup1[i].unsqueeze(0)
#         dist_norm = (current - lookup2).norm(dim=1, p=2)
#         dist_norm = dist_norm.detach().cpu().numpy()
#         dist[key1] = {}
#         for j, key2 in enumerate(keys2):
#             dist[key1][key2] = dist_norm[j]
#     return dist


# labels_file = '/root/CLEAN/app/data/labels.pkl'
# model_lookup_file = '/root/CLEAN/app/data/model_lookup.pkl'

# with open (labels_file, 'rb') as file:
#     labels = pickle.load(file)

# one = 0
# zero = 0
# for i in labels:
#     if i == '1':
#         one = one +1
#     else:
#         zero = zero + 1
# # print(one, zero)    # 128290 127647

# new_labels = []
# for i in range(1,one+1):
#     new_labels.append(i)
# for j in range(1, zero+1):
#     new_labels.append(-j)

# # labels = new_labels[:100] + new_labels[-100:]
# labels = new_labels

# with open (model_lookup_file, 'rb') as file:
#     new_model_lookup = pickle.load(file)
# # model_lookup = np.r_[new_model_lookup[:100],new_model_lookup[-100:]]
# # pickle.dump(model_lookup, open('./data/distance_map/' + 'Emb' + '.pkl', 'wb'))
# model_lookup = new_model_lookup

# # 计算labels的个数，及model_lookup的维度
# # print(f'labels 的长度：{len(labels)}')     # 255937
# # print(f'model_lookup 的维度：{model_lookup.shape}')    # (255937, 1280)

# # 将NumPy数组转换为Tensor
# model_lookup = torch.from_numpy(model_lookup)
# model_dist = dist_map_helper(labels, model_lookup, labels, model_lookup)
# pickle.dump(model_dist, open('./data/distance_map/' + 'model_dist' + '.pkl', 'wb'))



#############################################
##### 2 ###### 600万数据 制作 model_lookup 及 labels
#############################################
# # 获取Uniprot的 emb
# import h5py
# import torch
# per_protein = h5py.File('/root/CLEAN/app/data/inputs/CLearning/training/per-protein.h5', 'r')

# # 获取 rxn_emb
# file = '/root/CLEAN/app/data/inputs/CLearning/training/rxn_embeddings_dict.pkl'
# with open(file,'rb') as f:
#     data_rxn = pickle.load(f) 

# # 获取 train 的数据
# file = '/root/CLEAN/app/data/inputs/CLearning/training/train.pkl'
# with open(file,'rb') as f:
#     data = pickle.load(f)       # DataFrame ['RHEA_ID', 'rxn_smiles', 'ec', 'uniprot', 'tag']

# data_list = data.values.tolist()


# labels = []
# for ind, item in tqdm(enumerate(data_list),total=len(data_list)):
#     model_lookup = []
#     # 获取 rxn_emb
#     rxn = item[1]
#     rxn_emb = data_rxn[rxn]

#     # 获取 seq_emb
#     seq_id = item[3]
#     seq_emb = per_protein[seq_id][:]

#     # 拼接 rxn_seq
#     model_lookup.append(list(rxn_emb)+list(seq_emb))  

#     # 统计 labels
#     labels.append(item[4])

#     data = {
#         'labels':item[4],
#         'emb':torch.Tensor(model_lookup)
#     }
#     output_file_1 = '/mnt/data/CL_train_model_lookup/train_model_lookup_'+ str(ind) +'.pt'
#     torch.save(data, output_file_1)

#     # if ind % 1000000 == 0 and ind != 0: 
#     #     output_file_1 = '/mnt/data/CL_train_model_lookup/train_model_lookup_'+ str(ind) +'.pkl'
#     #     with open(output_file_1, 'wb') as file:
#     #         # file.write(model_lookup)
#     #         pickle.dump(np.array(model_lookup), file)
#     #     model_lookup = []

#     # if ind == len(data_list)-1:
#     #     output_file_1 = '/mnt/data/CL_train_model_lookup/train_model_lookup_'+ str(ind) +'.pkl'
#     #     with open(output_file_1, 'wb') as file:
#     #         # file.write(model_lookup)
#     #         pickle.dump(np.array(model_lookup), file)

# output_file_2 = '/root/CLEAN/app/data/inputs/CLearning/training/train_labels.pkl'
# with open(output_file_2, 'wb') as file:
#     pickle.dump(labels, file)

#############################################
##### 3 ###### 600万数据 制作 model_lookup 及 labels [数组格式]
############################################
# import random
# import torch

# P_Emb_list = []
# for i in tqdm(range(26000),total=26000):
#     ind = random.randint(0, 178940)
#     data = torch.load('/mnt/data/CL_train_model_lookup/train_model_lookup_'+ str(ind) +'.pt')
#     emb = data['emb']
#     emb = emb.numpy()
#     P_Emb_list.append(emb)
    
# P_Emb = np.concatenate(P_Emb_list,axis=0)
# print(P_Emb.shape)    # torch.Size([26000, 1280])

# N_Emb_list = []
# for i in tqdm(range(974000),total=974000):
#     ind = random.randint(178941, 6766140)
#     data = torch.load('/mnt/data/CL_train_model_lookup/train_model_lookup_'+ str(ind) +'.pt')
#     emb = data['emb']
#     emb = emb.numpy()
#     N_Emb_list.append(emb)

# N_Emb = np.concatenate(N_Emb_list,axis=0)
# print(N_Emb.shape) 

# EMB = np.concatenate([P_Emb,N_Emb], axis=0)

# output_file ='/root/CLEAN/app/data/inputs/CLearning/training_1/train_1.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(EMB, file)

# # labels
# labels = []
# for i in range(1000000):
#     labels.append(1)
#     if i >= 26000:
#         labels.append(0)
# output_file_2 = '/root/CLEAN/app/data/inputs/CLearning/training_1/train_labels.pkl'
# with open(output_file_2, 'wb') as file:
#     pickle.dump(labels, file)



#############################################
##### 4 ###### 制作 600万数据 model_lookup 及 labels [Tensor格式]
############################################
# import random
# import torch

# P_Emb_list = []
# for i in tqdm(range(26000),total=26000):
#     ind = random.randint(0, 178940)
#     data = torch.load('/mnt/data/CL_train_model_lookup/train_model_lookup_'+ str(ind) +'.pt')
#     emb = data['emb']
#     P_Emb_list.append(emb)
    
# P_Emb = torch.cat(P_Emb_list, dim=0)
# print(P_Emb.shape)    # torch.Size([26000, 1280])

# N_Emb_list = []
# for i in tqdm(range(974000),total=974000):
#     ind = random.randint(178941, 6766140)
#     data = torch.load('/mnt/data/CL_train_model_lookup/train_model_lookup_'+ str(ind) +'.pt')
#     emb = data['emb']
#     N_Emb_list.append(emb)

# N_Emb = torch.cat(N_Emb_list, dim=0)
# print(N_Emb.shape) 

# EMB = torch.cat([P_Emb,N_Emb], dim=0)
# print(EMB.shape)
# torch.save(EMB, '/root/CLEAN/app/data/inputs/CLearning/training_1/train_tensor_6.pt')


# # labels
# labels = []
# for i in range(1000000):
#     if i >= 26000:
#         labels.append(0)
#     else:
#         labels.append(1)
# output_file_2 = '/root/CLEAN/app/data/inputs/CLearning/training_6parts_6million/train_labels.pkl'
# with open(output_file_2, 'wb') as file:
#     pickle.dump(labels, file)
# print(len(labels))


#############################################################################################################################
#############################################################################################################################

#############################################
##### 5 ###### 预测rxn的EC  制作 model_lookup 及 labels [tensor格式]
#############################################
# rxn_file = '/root/CLEAN/app/data/inputs/rxn_pred_EC/rxn_embeddings_dict.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)

# samples_file = '/root/CLEAN/app/data/inputs/rxn_pred_EC/train_test.csv'
# with open(samples_file, 'r', encoding='utf-8') as f:
#     samples = csv.reader(f)
#     next(samples)
#     samples = list(samples)

# model_lookup_train = []
# model_lookup_test = []
# labels_train = []
# labels_test = []
# for ind, item in tqdm(enumerate(samples),total=len(samples)):   
#     # 检索 rxn 的 Emb 
#     if item[3] in rxns.keys():
#         rxn = rxns[item[3]] 
#     else:
#         # print(f'找不到rxn{ind}:{samples[ind]}')
#         continue
#     if item[1] =='train':
#         model_lookup_train.append(rxn)
#         labels_train.append(item[0])
#     else:
#         model_lookup_test.append(rxn)
#         labels_test.append(item[0])

# print(len(model_lookup_train))  # 220195
# print(len(model_lookup_test))    # 24232

# output_file = '/root/CLEAN/app/data/inputs/rxn_pred_EC/model_lookup_train.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(np.array(model_lookup_train), file)

# output_file = '/root/CLEAN/app/data/inputs/rxn_pred_EC/model_lookup_test.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(np.array(model_lookup_test), file)

# output_file = '/root/CLEAN/app/data/inputs/rxn_pred_EC/labels_train.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(labels_train, file)

# output_file = '/root/CLEAN/app/data/inputs/rxn_pred_EC/labels_test.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(labels_test, file)



#############################################
##### 5 ###### 预测rxn的EC  制作 esm_emb [tensor格式] [字典格式]
#############################################

# rxn_file = '/root/CLEAN/app/data/inputs/rxn_pred_EC/rxn_embeddings_dict.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)

# samples_file = '/root/CLEAN/app/data/inputs/rxn_pred_EC/train_test.csv'
# with open(samples_file, 'r', encoding='utf-8') as f:
#     samples = csv.reader(f)
#     next(samples)
#     samples = list(samples)

# model_lookup_train = {}
# model_lookup_test = {}
# labels_train = []
# labels_test = []
# for ind, item in tqdm(enumerate(samples),total=len(samples)):    
#     # 检索 rxn 的 Emb 
#     if item[3] in rxns.keys():
#         rxn = rxns[item[3]] 
#         rxn = torch.Tensor(rxn)
#     else:
#         # print(f'找不到rxn{ind}:{samples[ind]}')
#         continue
#     if item[1] =='train':
#         if item[0] in model_lookup_train.keys():
#             model_lookup_train[item[0]].append(rxn)
#         else:
#             temp_list = []
#             temp_list.append(rxn)
#             model_lookup_train[item[0]] = temp_list

#         labels_train.append(item[0])

#     else:
#         if item[0] in model_lookup_test.keys():
#             model_lookup_test[item[0]].append(rxn)
#         else:
#             temp_list = []
#             temp_list.append(rxn)
#             model_lookup_test[item[0]] = temp_list

#         labels_test.append(item[0])

# print(type(model_lookup_train['2.3.-.-']))
# print(len(model_lookup_train))  # 220195
# print(len(model_lookup_test))   # 24232

# output_file = '/root/CLEAN/app/data/inputs/rxn_pred_EC/model_lookup_train_dict.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(model_lookup_train, file)

# output_file = '/root/CLEAN/app/data/inputs/rxn_pred_EC/model_lookup_test_dict.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(model_lookup_test, file)

# output_file = '/root/CLEAN/app/data/inputs/rxn_pred_EC/labels_train.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(labels_train, file)

# output_file = '/root/CLEAN/app/data/inputs/rxn_pred_EC/labels_test.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(labels_test, file)

#############################################
########### 预测rxn的EC 制作 model_dist
#############################################        
# import numpy as np
# import torch
# import pickle

# def dist_map_helper(keys1, lookup1, keys2, lookup2):
#     dist = {}
#     for i, key1 in tqdm(enumerate(keys1), total=len(keys1)):
#         current = lookup1[i].unsqueeze(0)
#         dist_norm = (current - lookup2).norm(dim=1, p=2)
#         dist_norm = dist_norm.detach().cpu().numpy()
#         dist[key1] = {}
#         for j, key2 in enumerate(keys2):
#             dist[key1][key2] = dist_norm[j]
#     return dist


# labels_file = '/root/CLEAN/app/data/inputs/rxn_pred_EC/labels_train.pkl'
# model_lookup_file = '/root/CLEAN/app/data/inputs/rxn_pred_EC/model_lookup_train.pkl'

# with open (labels_file, 'rb') as file:
#     labels = pickle.load(file)

# with open (model_lookup_file, 'rb') as file:
#     model_lookup = pickle.load(file)

# # 将NumPy数组转换为Tensor
# model_lookup = torch.from_numpy(model_lookup)
# model_dist = dist_map_helper(labels, model_lookup, labels, model_lookup)

# file = '/root/CLEAN/app/data/inputs/rxn_pred_EC/model_dist.pkl'
# with open(file, 'wb') as f:
#     pickle.dump(model_dist, f)

'''
测试为什么neg的shape为([])

file = '/root/CLEAN/app/data/inputs/rxn_pred_EC/model_dist.pkl'
with open(file, 'rb') as f:
    model_dist = pickle.load( f)

neg_ec = '4.99.-.-'
neg = model_dist[neg_ec]

# print(neg)

file = '/root/CLEAN/app/data/inputs/rxn_pred_EC/model_lookup_train_dict.pkl'

with open (file, 'rb') as file:
    esm_emb = pickle.load(file)
print(len(esm_emb[neg_ec]))
for i in esm_emb[neg_ec]:
    print(i.size())
        # print(i)


# import random
# for i in range(10):
#     neg = random.choice(esm_emb[neg_ec])
#     print(neg.shape)

'''

################################################################################################################
########################### 新数据，记录loss随着epoch增加，以及不同learning rate的变化
################################################################################################################


###############################
### 预测rxn的EC  制作 model_lookup 及 labels [tensor格式]
###############################

# ### train
# rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/rxn_ec_emb.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)

# samples_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/train_augmented.csv'
# with open(samples_file, 'r', encoding='utf-8') as f:
#     samples = csv.reader(f)
#     next(samples)
#     train = list(samples)

# model_lookup_train = []
# labels_train = []
# for ind, item in tqdm(enumerate(train),total=len(train)):   
#     # 检索 rxn 的 Emb 
#     if item[0] in rxns.keys():
#         rxn = rxns[item[0]] 
#         model_lookup_train.append(rxn)
#         labels_train.append(item[2])
#     else:
#         # print(f'找不到rxn{ind}:{samples[ind]}')
#         continue

# print(len(model_lookup_train))  # 277860


# ## test
# rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/rxn_ec_emb_test.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)

# samples_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/test.csv'
# with open(samples_file, 'r', encoding='utf-8') as f:
#     samples = csv.reader(f)
#     next(samples)
#     test = list(samples)

# model_lookup_test = []
# labels_test = []

# for ind, item in tqdm(enumerate(test),total=len(test)):   
#     # 检索 rxn 的 Emb 
#     if item[0] in rxns.keys():
#         rxn = rxns[item[0]] 
#         model_lookup_test.append(rxn)
#         labels_test.append(item[2])
#     else:
#         # print(f'找不到rxn{ind}:{test[ind]}')
#         continue

# print(len(model_lookup_test))    # 


# # output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/model_lookup_train.pkl'
# # with open(output_file, 'wb') as file:
# #     pickle.dump(np.array(model_lookup_train), file)

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/model_lookup_test.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(np.array(model_lookup_test), file)

# # output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/labels_train.pkl'
# # with open(output_file, 'wb') as file:
# #     pickle.dump(labels_train, file)

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/labels_test.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(labels_test, file)



#############################################
#####  预测rxn的EC  制作 esm_emb [tensor格式] [字典格式]
#############################################
# ### train
# rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/rxn_ec_emb.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)

# samples_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/train_augmented.csv'
# with open(samples_file, 'r', encoding='utf-8') as f:
#     samples = csv.reader(f)
#     next(samples)
#     samples = list(samples)

# model_lookup_train = {}
# labels_train = []
# for ind, item in tqdm(enumerate(samples),total=len(samples)):    
#     # 检索 rxn 的 Emb 
#     if item[0] in rxns.keys():
#         rxn = rxns[item[0]] 
#         rxn = torch.Tensor(rxn)
#     else:
#         # print(f'找不到rxn{ind}:{samples[ind]}')
#         continue

#     if item[2] in model_lookup_train.keys():
#         model_lookup_train[item[2]].append(rxn)
#     else:
#         temp_list = []
#         temp_list.append(rxn)
#         model_lookup_train[item[2]] = temp_list

#     labels_train.append(item[2])


# print(type(model_lookup_train['2.3.1.199']))
# print(len(model_lookup_train))  # unique_ec:914
# print(len(labels_train))    # 277860


# # ## test
# # rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/rxn_ec_emb_test.pkl'
# # with open(rxn_file,'rb') as f:
# #     rxns = pickle.load(f)

# # samples_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/train_augmented.csv'
# # with open(samples_file, 'r', encoding='utf-8') as f:
# #     samples = csv.reader(f)
# #     next(samples)
# #     samples = list(samples)

# # model_lookup_test = {}
# # labels_test = []
# # for ind, item in tqdm(enumerate(samples),total=len(samples)):    
# #     # 检索 rxn 的 Emb 
# #     if item[0] in rxns.keys():
# #         rxn = rxns[item[0]] 
# #         rxn = torch.Tensor(rxn)
# #     else:
# #         # print(f'找不到rxn{ind}:{samples[ind]}')
# #         continue

# #     if item[2] in model_lookup_test.keys():
# #         model_lookup_test[item[2]].append(rxn)
# #     else:
# #         temp_list = []
# #         temp_list.append(rxn)
# #         model_lookup_test[item[2]] = temp_list

# #     labels_test.append(item[2])


# # print(type(model_lookup_test['2.3.1.199']))
# # print(len(model_lookup_test))  # unique_ec:
# # print(len(labels_test))


# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/model_lookup_train_dict.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(model_lookup_train, file)

# # output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/model_lookup_test_dict.pkl'
# # with open(output_file, 'wb') as file:
# #     pickle.dump(model_lookup_test, file)

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/labels_train.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(labels_train, file)

# # output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/labels_test.pkl'
# # with open(output_file, 'wb') as file:
# #     pickle.dump(labels_test, file)


#############################################
########### 预测rxn的EC 制作 model_dist
#############################################        
# import numpy as np
# import torch
# import pickle

# def dist_map_helper(keys1, lookup1, keys2, lookup2):
#     dist = {}
#     for i, key1 in tqdm(enumerate(keys1), total=len(keys1)):
#         current = lookup1[i].unsqueeze(0)
#         dist_norm = (current - lookup2).norm(dim=1, p=2)
#         dist_norm = dist_norm.detach().cpu().numpy()
#         dist[key1] = {}
#         for j, key2 in enumerate(keys2):
#             dist[key1][key2] = dist_norm[j]
#     return dist


# labels_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/labels_train.pkl'
# model_lookup_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/model_lookup_train.pkl'

# with open (labels_file, 'rb') as file:
#     labels = pickle.load(file)

# with open (model_lookup_file, 'rb') as file:
#     model_lookup = pickle.load(file)

# # 将NumPy数组转换为Tensor
# model_lookup = torch.from_numpy(model_lookup)
# model_dist = dist_map_helper(labels, model_lookup, labels, model_lookup)

# file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/model_dist_train.pkl'
# with open(file, 'wb') as f:
#     pickle.dump(model_dist, f)



##################################
##########  lr loss plot
##################################

import matplotlib.pyplot as plt

# 假设你的数据是这样的
# loss_3 = [[1, 0.93423], [2, 0.838423], [3, 0.7321], [4, 0.68765],[5, 0.54345], [6, 0.42379]]
# loss_4 = [[1, 0.83423], [2, 0.78423], [3, 0.721], [4, 0.565],[5, 0.545], [6, 0.32379]]
# loss_5 = [[1, 0.73423], [2, 0.68423], [3, 0.521], [4, 0.3],[5, 0.245], [6, 0.12379]]

with open('/root/CLEAN/app/results/inputs/pred_rxn_EC1_2024319_nolabel/random_train_loss_10-4.pkl', 'rb') as file:
    loss_3 = pickle.load(file)
with open('/root/CLEAN/app/results/inputs/pred_rxn_EC2_2024319_nolabel/random_train_loss_10-4.pkl', 'rb') as file:
    loss_4 = pickle.load(file)
with open('/root/CLEAN/app/results/inputs/pred_rxn_EC3_2024314_nolabel/random_train_loss_10-4.pkl', 'rb') as file:
    loss_5 = pickle.load(file)

print(loss_3[:10])
print(loss_4[:10])
print(loss_5[:10])

# 提取 epoch 和 loss
epochs3, losses3 = zip(*loss_3)
epochs4, losses4 = zip(*loss_4)
epochs5, losses5 = zip(*loss_5)

# 创建折线图
# plt.plot(epochs3, losses3, label='lr = 5*10-3', color='black', linestyle='-', linewidth=1)
# plt.plot(epochs4, losses4, label='lr = 5*10-4', color='black', linestyle=':')
# plt.plot(epochs5, losses5, label='lr = 5*10-5', color='black', linestyle='-.', linewidth=1)
plt.plot(epochs3, losses3, label='EC1', linewidth=1)
plt.plot(epochs4, losses4, label='EC12',linewidth=1)
plt.plot(epochs5, losses5, label='EC123',linewidth=1)

# 添加标签和标题
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.title('Loss over Epochs')

# 显示图例
plt.legend()

plt.savefig('/root/CLEAN/app/results/inputs/pred_rxn_EC2_2024319_nolabel/line_plot.png')
# 显示图形
plt.show()


# #############################################
# #####  预测rxn的EC  制作 esm_emb [tensor格式] [字典格式]   更详细的EC，不带*
# #############################################
# ### train
# rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/rxn_ec_emb.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)

# samples_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/train_augmented.csv'
# with open(samples_file, 'r', encoding='utf-8') as f:
#     samples = csv.reader(f)
#     next(samples)
#     samples = list(samples)

# model_lookup_train = {}
# labels_train = []
# for ind, item in tqdm(enumerate(samples),total=len(samples)):    
#     # 检索 rxn 的 Emb 
#     if item[0] in rxns.keys():
#         rxn = rxns[item[0]] 
#         rxn = torch.Tensor(rxn)
#     else:
#         # print(f'找不到rxn{ind}:{samples[ind]}')
#         continue

#     if item[2] in model_lookup_train.keys():
#         model_lookup_train[item[1]].append(rxn)
#     else:
#         temp_list = []
#         temp_list.append(rxn)
#         model_lookup_train[item[1]] = temp_list

#     labels_train.append(item[1])


# print(type(model_lookup_train['2.3.1.199']))
# print(len(model_lookup_train))  # unique_ec: 6096
# print(len(labels_train))    # 277860


# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/model_lookup_train_dict_nostar.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(model_lookup_train, file)

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/labels_train_nostar.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(labels_train, file)


# ######################################
# ########  EC不带*   test的model_lookup
# ######################################
# ## test
# rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/rxn_ec_emb_test.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)

# samples_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/test.csv'
# with open(samples_file, 'r', encoding='utf-8') as f:
#     samples = csv.reader(f)
#     next(samples)
#     test = list(samples)

# model_lookup_test = []
# labels_test = []

# for ind, item in tqdm(enumerate(test),total=len(test)):   
#     # 检索 rxn 的 Emb 
#     if item[0] in rxns.keys():
#         rxn = rxns[item[0]] 
#         model_lookup_test.append(rxn)
#         labels_test.append(item[1])
#     else:
#         # print(f'找不到rxn{ind}:{test[ind]}')
#         continue

# print(len(model_lookup_test))    # 6650

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/nostar_model_lookup_test.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(np.array(model_lookup_test), file)


# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/nostar_labels_test.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(labels_test, file)


# #############################################
# #####  预测rxn的EC  制作 esm_emb [tensor格式] [字典格式]   除去EC带*的数据
# #############################################
# ### train
# rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/rxn_ec_emb.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)

# samples_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/train_augmented.csv'
# with open(samples_file, 'r', encoding='utf-8') as f:
#     samples = csv.reader(f)
#     next(samples)
#     samples = list(samples)

# model_lookup_train = {}
# labels_train = []
# for ind, item in tqdm(enumerate(samples),total=len(samples)):   
#     if '*' in item[2]:
#         continue 

#     # 检索 rxn 的 Emb 
#     if item[0] in rxns.keys():
#         rxn = rxns[item[0]] 
#         rxn = torch.Tensor(rxn)
#     else:
#         # print(f'找不到rxn{ind}:{samples[ind]}')
#         continue

#     if item[2] in model_lookup_train.keys():
#         model_lookup_train[item[2]].append(rxn)
#     else:
#         temp_list = []
#         temp_list.append(rxn)
#         model_lookup_train[item[2]] = temp_list

#     labels_train.append(item[2])


# print(type(model_lookup_train['2.3.1.199']))
# print(len(model_lookup_train))  # unique_ec:718  
# print(len(labels_train))    # 205180



# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/removestar_model_lookup_train_dict.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(model_lookup_train, file)

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/removestar_labels_train.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(labels_train, file)


# ######################################
# ########  去除带 * 的EC   train、test的model_lookup
# ######################################

# ### train
# rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/rxn_ec_emb.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)

# samples_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/train_augmented.csv'
# with open(samples_file, 'r', encoding='utf-8') as f:
#     samples = csv.reader(f)
#     next(samples)
#     train = list(samples)

# model_lookup_train = []
# labels_train = []
# for ind, item in tqdm(enumerate(train),total=len(train)):  
#     if '*' in item[2]:
#         continue 
#     # 检索 rxn 的 Emb 
#     if item[0] in rxns.keys():
#         rxn = rxns[item[0]] 
#         model_lookup_train.append(rxn)
#         labels_train.append(item[2])
#     else:
#         # print(f'找不到rxn{ind}:{test[ind]}')
#         continue

# print(len(model_lookup_train))  # 205180



# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/removestar_model_lookup_train.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(np.array(model_lookup_train), file)


# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/removestar_labels_train.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(labels_train, file)



# ## test
# rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/rxn_ec_emb_test.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)

# samples_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/test.csv'
# with open(samples_file, 'r', encoding='utf-8') as f:
#     samples = csv.reader(f)
#     next(samples)
#     test = list(samples)

# model_lookup_test = []
# labels_test = []

# for ind, item in tqdm(enumerate(test),total=len(test)):  
#     if '*' in item[2]:
#         continue 
#     # 检索 rxn 的 Emb 
#     if item[0] in rxns.keys():
#         rxn = rxns[item[0]] 
#         model_lookup_test.append(rxn)
#         labels_test.append(item[2])
#     else:
#         # print(f'找不到rxn{ind}:{test[ind]}')
#         continue

# print(len(model_lookup_test))    # 4934

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/removestar_model_lookup_test.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(np.array(model_lookup_test), file)


# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/removestar_labels_test.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(labels_test, file)


###############################################################################################################################
###############################################################################################################################
######  数据集：只取EC的前三位作为标签  model_lookup train and test
#################################################

# ### train
# rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/rxn_ec_emb.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)

# samples_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/train_augmented.csv'
# with open(samples_file, 'r', encoding='utf-8') as f:
#     samples = csv.reader(f)
#     next(samples)
#     train = list(samples)

# model_lookup_train = []
# labels_train = []
# for ind, item in tqdm(enumerate(train),total=len(train)):   
#     # 检索 rxn 的 Emb 
#     if item[0] in rxns.keys():
#         rxn = rxns[item[0]] 
#         model_lookup_train.append(rxn)
#         label_list = item[1].split('.')[:3]
#         label = '.'.join(label_list)
#         labels_train.append(label)
#     else:
#         # print(f'找不到rxn{ind}:{samples[ind]}')
#         continue

# print(len(model_lookup_train))  # 277860


# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_202437/model_lookup_train.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(np.array(model_lookup_train), file)

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_202437/labels_train.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(labels_train, file)



# ### test
# rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/rxn_ec_emb_test.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)

# samples_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/test.csv'
# with open(samples_file, 'r', encoding='utf-8') as f:
#     samples = csv.reader(f)
#     next(samples)
#     test = list(samples)

# model_lookup_test = []
# labels_test = []
# for ind, item in tqdm(enumerate(test),total=len(test)):   
#     # 检索 rxn 的 Emb 
#     if item[0] in rxns.keys():
#         rxn = rxns[item[0]] 
#         model_lookup_test.append(rxn)
#         label_list = item[1].split('.')[:3]
#         label = '.'.join(label_list)
#         labels_test.append(label)
#     else:
#         # print(f'找不到rxn{ind}:{test[ind]}')
#         continue

# print(len(model_lookup_test))    # 6650

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_202437/model_lookup_test.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(np.array(model_lookup_test), file)

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_202437/labels_test.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(labels_test, file)


# #############################################
# #####  数据集：只取EC的前三位作为标签  制作 esm_emb [tensor格式] [字典格式]   
# #############################################
# ### train
# rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/rxn_ec_emb.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)

# samples_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC_2024229/train_augmented.csv'
# with open(samples_file, 'r', encoding='utf-8') as f:
#     samples = csv.reader(f)
#     next(samples)
#     samples = list(samples)

# model_lookup_train = {}
# for ind, item in tqdm(enumerate(samples),total=len(samples)):    
#     # 检索 rxn 的 Emb 
#     if item[0] in rxns.keys():
#         rxn = rxns[item[0]] 
#         rxn = torch.Tensor(rxn)
#     else:
#         # print(f'找不到rxn{ind}:{samples[ind]}')
#         continue

#     label_list = item[1].split('.')[:3]
#     label = '.'.join(label_list)
    
#     if label in model_lookup_train.keys():
#         model_lookup_train[label].append(rxn)
#     else:
#         temp_list = []
#         temp_list.append(rxn)
        
#         model_lookup_train[label] = temp_list


# print(len(model_lookup_train))  # unique_ec: 307


# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_202437/esm_emb_dict.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(model_lookup_train, file)


#############################################
########### 数据集：只取EC的前三位作为标签  制作 model_dist
#############################################        
# import numpy as np
# import torch
# import pickle

# def dist_map_helper(keys1, lookup1, keys2, lookup2):
#     dist = {}
#     for i, key1 in tqdm(enumerate(keys1), total=len(keys1)):
#         current = lookup1[i].unsqueeze(0)
#         dist_norm = (current - lookup2).norm(dim=1, p=2)
#         dist_norm = dist_norm.detach().cpu().numpy()
#         dist[key1] = {}
#         for j, key2 in enumerate(keys2):
#             dist[key1][key2] = dist_norm[j]
#     return dist


# labels_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_202437/labels_train.pkl'
# model_lookup_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_202437/model_lookup_train.pkl'

# with open (labels_file, 'rb') as file:
#     labels = pickle.load(file)

# with open (model_lookup_file, 'rb') as file:
#     model_lookup = pickle.load(file)

# # 将NumPy数组转换为Tensor
# model_lookup = torch.from_numpy(model_lookup)
# model_dist = dist_map_helper(labels, model_lookup, labels, model_lookup)

# file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_202437/model_dist_train.pkl'
# with open(file, 'wb') as f:
#     pickle.dump(model_dist, f)


###############################################################################################################################
###############################################################################################################################
######  数据集：只取EC的前三位作为标签(rxn 中无标签版本)  model_lookup train and test
#################################################

# ### train
# rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel/rxn_ec_emb.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)

# samples_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel/train_augmented.csv'
# with open(samples_file, 'r', encoding='utf-8') as f:
#     samples = csv.reader(f)
#     next(samples)
#     train = list(samples)

# model_lookup_train = []
# labels_train = []
# for ind, item in tqdm(enumerate(train),total=len(train)):   
#     # 检索 rxn 的 Emb 
#     if item[0] in rxns.keys():
#         rxn = rxns[item[0]] 
#         model_lookup_train.append(rxn)
#         labels_train.append(item[1])
#     else:
#         # print(f'找不到rxn{ind}:{samples[ind]}')
#         continue

# print(len(model_lookup_train))  # 166918


# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel/model_lookup_train.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(np.array(model_lookup_train), file)

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel/labels_train.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(labels_train, file)



# ### test
# rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel/rxn_ec_emb.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)

# samples_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel/test_augmented.csv'
# with open(samples_file, 'r', encoding='utf-8') as f:
#     samples = csv.reader(f)
#     next(samples)
#     test = list(samples)

# model_lookup_test = []
# labels_test = []
# for ind, item in tqdm(enumerate(test),total=len(test)):   
#     # 检索 rxn 的 Emb 
#     if item[0] in rxns.keys():
#         rxn = rxns[item[0]] 
#         model_lookup_test.append(rxn)
#         labels_test.append(item[1])
#     else:
#         # print(f'找不到rxn{ind}:{test[ind]}')
#         continue

# print(len(model_lookup_test))    # 18816

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel/model_lookup_test.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(np.array(model_lookup_test), file)

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel/labels_test.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(labels_test, file)


# #############################################
# #####  数据集：只取EC的前三位作为标签  制作 esm_emb [tensor格式] [字典格式]   
# #############################################
# ### train
# rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel/rxn_ec_emb.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)

# samples_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel/train_augmented.csv'
# with open(samples_file, 'r', encoding='utf-8') as f:
#     samples = csv.reader(f)
#     next(samples)
#     samples = list(samples)

# model_lookup_train = {}
# for ind, item in tqdm(enumerate(samples),total=len(samples)):    
#     # 检索 rxn 的 Emb 
#     if item[0] in rxns.keys():
#         rxn = rxns[item[0]] 
#         rxn = torch.Tensor(rxn)
#     else:
#         # print(f'找不到rxn{ind}:{samples[ind]}')
#         continue

    
    
#     if item[1] in model_lookup_train.keys():
#         model_lookup_train[item[1]].append(rxn)
#     else:
#         temp_list = []
#         temp_list.append(rxn)
        
#         model_lookup_train[item[1]] = temp_list


# print(len(model_lookup_train))  # unique_ec: 176


# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel/esm_emb_dict.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(model_lookup_train, file)


#############################################
########### 数据集：只取EC的前三位作为标签  制作 model_dist
#############################################        
# import numpy as np
# import torch
# import pickle

# def dist_map_helper(keys1, lookup1, keys2, lookup2):
#     dist = {}
#     for i, key1 in tqdm(enumerate(keys1), total=len(keys1)):
#         current = lookup1[i].unsqueeze(0)
#         dist_norm = (current - lookup2).norm(dim=1, p=2)
#         dist_norm = dist_norm.detach().cpu().numpy()
#         dist[key1] = {}
#         for j, key2 in enumerate(keys2):
#             dist[key1][key2] = dist_norm[j]
#     return dist


# labels_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel/labels_train.pkl'
# model_lookup_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel/model_lookup_train.pkl'

# with open (labels_file, 'rb') as file:
#     labels = pickle.load(file)

# with open (model_lookup_file, 'rb') as file:
#     model_lookup = pickle.load(file)

# # 将NumPy数组转换为Tensor
# model_lookup = torch.from_numpy(model_lookup)
# model_dist = dist_map_helper(labels, model_lookup, labels, model_lookup)

# file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel/model_dist_train.pkl'
# with open(file, 'wb') as f:
#     pickle.dump(model_dist, f)


###############################################################################################################################
# ##############################################################################################################################
# #####  数据集：只取EC的前三位作为标签(rxn 中无标签版本,训练集的每个EC只随机选取100个)  model_lookup train and test
# ################################################

# ### train
# import random

# rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel/esm_emb_dict.pkl'
# with open(rxn_file,'rb') as f:
#     esm_emb = pickle.load(f)

# model_lookup_train = []
# labels_train = []
# new_esm_emb = {}
# for key, value in esm_emb.items():
#     temp_list = random.choices(value, k=min(100, len(value)))
#     model_lookup_array = []
#     for i in temp_list:
#         x = np.array(i)
#         model_lookup_array.append(x)

#     model_lookup_train.extend(model_lookup_array)
#     for i in range(len(temp_list)):
#         labels_train.append(key)
#     new_esm_emb[key] = temp_list

# print(len(esm_emb)) # 176
# print(len(model_lookup_train))  # 14379
# print(len(labels_train))    # 14379
# print(len(new_esm_emb))    # 176

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel_100/model_lookup_train.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(np.array(model_lookup_train), file)

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel_100/labels_train.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(labels_train, file)

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel_100/esm_emb_dict.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(new_esm_emb, file)




#############################################
########### 数据集：只取EC的前三位作为标签  制作 model_dist
#############################################        
# import numpy as np
# import torch
# import pickle

# def dist_map_helper(keys1, lookup1, keys2, lookup2):
#     dist = {}
#     for i, key1 in tqdm(enumerate(keys1), total=len(keys1)):
#         current = lookup1[i].unsqueeze(0)
#         dist_norm = (current - lookup2).norm(dim=1, p=2)
#         dist_norm = dist_norm.detach().cpu().numpy()
#         dist[key1] = {}
#         for j, key2 in enumerate(keys2):
#             dist[key1][key2] = dist_norm[j]
#     return dist


# labels_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel_100/labels_train.pkl'
# model_lookup_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel_100/model_lookup_train.pkl'

# with open (labels_file, 'rb') as file:
#     labels = pickle.load(file)

# with open (model_lookup_file, 'rb') as file:
#     model_lookup = pickle.load(file)

# # 将NumPy数组转换为Tensor
# model_lookup = torch.from_numpy(model_lookup)
# model_dist = dist_map_helper(labels, model_lookup, labels, model_lookup)

# file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel_100/model_dist_train.pkl'
# with open(file, 'wb') as f:
#     pickle.dump(model_dist, f)


###############################################################################################################################
# ##############################################################################################################################
# #####  数据集：只取EC的前三位作为标签(rxn 中无标签版本,rxn_embedding 采用drfp的方式)  model_lookup train and test
# ################################################

# ### train
# rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel/train_rxn_embedding.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)
# print(type(rxns))   # list
# print(type(rxns[0]))    # numpy.ndarray
# print(len(rxns))    # 166918

# samples_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel/train_augmented.csv'
# data = pd.read_csv(samples_file)
# labels_train = data['ec3'].values.tolist()
# print(len(labels_train)) # 166918

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_ec3_20242318_nolabel_drfp/model_lookup_train.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(np.array(rxns), file)

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_ec3_20242318_nolabel_drfp/labels_train.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(labels_train, file)


### test
# rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel/test_rxn_embedding.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)
# print(type(rxns))   # list
# print(type(rxns[0]))    # numpy.ndarray
# print(len(rxns))    # 18816
# print(rxns[:5])

# samples_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel/test_augmented.csv'
# data = pd.read_csv(samples_file)
# labels_test = data['ec3'].values.tolist()
# print(len(labels_test))   # 18816

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_ec3_20242318_nolabel_drfp/model_lookup_test.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(np.array(rxns), file)

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_ec3_20242318_nolabel_drfp/labels_test.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(labels_test, file)


 #############################################
# #####  数据集：只取EC的前三位作为标签  制作 esm_emb [tensor格式] [字典格式]   
# #############################################
# ### train
# rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel/train_rxn_embedding.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)

# samples_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC3_2024314_nolabel/train_augmented.csv'
# with open(samples_file, 'r', encoding='utf-8') as f:
#     samples = csv.reader(f)
#     next(samples)
#     samples = list(samples)

# model_lookup_train = {}
# for ind, item in tqdm(enumerate(samples),total=len(samples)):    
#     # 检索 rxn 的 Emb 
#     rxn = torch.Tensor(rxns[ind])

#     if item[1] in model_lookup_train.keys():
#         model_lookup_train[item[1]].append(rxn)
#     else:
#         temp_list = []
#         temp_list.append(rxn)
        
#         model_lookup_train[item[1]] = temp_list

# print(len(model_lookup_train))  # unique_ec: 176

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_ec3_20242318_nolabel_drfp/esm_emb_dict.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(model_lookup_train, file)


#############################################
########### 数据集：只取EC的前三位作为标签  制作 model_dist
#############################################        
# import numpy as np
# import torch
# import pickle

# def dist_map_helper(keys1, lookup1, keys2, lookup2):
#     dist = {}
#     for i, key1 in tqdm(enumerate(keys1), total=len(keys1)):
#         current = lookup1[i].unsqueeze(0)
#         dist_norm = (current - lookup2).norm(dim=1, p=2)
#         dist_norm = dist_norm.detach().cpu().numpy()
#         dist[key1] = {}
#         for j, key2 in enumerate(keys2):
#             dist[key1][key2] = dist_norm[j]
#     return dist


# labels_file = '/root/CLEAN/app/data/inputs/pred_rxn_ec3_20242318_nolabel_drfp/labels_train.pkl'
# model_lookup_file = '/root/CLEAN/app/data/inputs/pred_rxn_ec3_20242318_nolabel_drfp/model_lookup_train.pkl'

# with open (labels_file, 'rb') as file:
#     labels = pickle.load(file)

# with open (model_lookup_file, 'rb') as file:
#     model_lookup = pickle.load(file)
# model_lookup = model_lookup.astype(float)
# print(model_lookup[:5])


# # 将NumPy数组转换为Tensor
# model_lookup = torch.from_numpy(model_lookup)
# model_dist = dist_map_helper(labels, model_lookup, labels, model_lookup)

# file = '/root/CLEAN/app/data/inputs/pred_rxn_ec3_20242318_nolabel_drfp/model_dist_train.pkl'
# with open(file, 'wb') as f:
#     pickle.dump(model_dist, f)


###############################################################################################################################
###############################################################################################################################
######  数据集：只取EC的前一位作为标签(rxn 中无标签版本)  model_lookup train and test
#################################################

# ### train
# rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC1_2024319_nolabel/rxn_ec_emb.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)

# samples_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC1_2024319_nolabel/train_augmented.csv'
# with open(samples_file, 'r', encoding='utf-8') as f:
#     samples = csv.reader(f)
#     next(samples)
#     train = list(samples)

# model_lookup_train = []
# labels_train = []
# for ind, item in tqdm(enumerate(train),total=len(train)):   
#     # 检索 rxn 的 Emb 
#     if item[0] in rxns.keys():
#         rxn = rxns[item[0]] 
#         model_lookup_train.append(rxn)
#         label = item[1].split('.')[0]
#         labels_train.append(label)
#     else:
#         # print(f'找不到rxn{ind}:{samples[ind]}')
#         continue

# print(len(model_lookup_train))  # 166918


# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC1_2024319_nolabel/model_lookup_train.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(np.array(model_lookup_train), file)

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC1_2024319_nolabel/labels_train.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(labels_train, file)



### test
# rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC1_2024319_nolabel/rxn_ec_emb.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)

# samples_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC1_2024319_nolabel/test_augmented.csv'
# with open(samples_file, 'r', encoding='utf-8') as f:
#     samples = csv.reader(f)
#     next(samples)
#     test = list(samples)

# model_lookup_test = []
# labels_test = []
# for ind, item in tqdm(enumerate(test),total=len(test)):   
#     # 检索 rxn 的 Emb 
#     if item[0] in rxns.keys():
#         rxn = rxns[item[0]] 
#         model_lookup_test.append(rxn)
#         label = item[1].split('.')[0]
#         labels_test.append(label)
#     else:
#         # print(f'找不到rxn{ind}:{test[ind]}')
#         continue

# print(len(model_lookup_test))    # 18816

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC1_2024319_nolabel/model_lookup_test.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(np.array(model_lookup_test), file)

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC1_2024319_nolabel/labels_test.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(labels_test, file)


# #############################################
# #####  数据集：只取EC的第一位作为标签  制作 esm_emb [tensor格式] [字典格式]   
# #############################################
# ### train
# rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC1_2024319_nolabel/rxn_ec_emb.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)

# samples_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC1_2024319_nolabel/train_augmented.csv'
# with open(samples_file, 'r', encoding='utf-8') as f:
#     samples = csv.reader(f)
#     next(samples)
#     samples = list(samples)

# model_lookup_train = {}
# for ind, item in tqdm(enumerate(samples),total=len(samples)):    
#     # 检索 rxn 的 Emb 
#     if item[0] in rxns.keys():
#         rxn = rxns[item[0]] 
#         rxn = torch.Tensor(rxn)
#     else:
#         # print(f'找不到rxn{ind}:{samples[ind]}')
#         continue

#     label = item[1].split('.')[0]

#     if label in model_lookup_train.keys():
#         model_lookup_train[label].append(rxn)
#     else:
#         temp_list = []
#         temp_list.append(rxn)
        
#         model_lookup_train[label] = temp_list


# print(len(model_lookup_train))  # unique_ec: 8


# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC1_2024319_nolabel/esm_emb_dict.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(model_lookup_train, file)


###############################################################################################################################
###############################################################################################################################
######  数据集：只取EC的前两位作为标签(rxn 中无标签版本)  model_lookup train and test
#################################################

# ### train
# rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC2_2024319_nolabel/rxn_ec_emb.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)

# samples_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC2_2024319_nolabel/train_augmented.csv'
# with open(samples_file, 'r', encoding='utf-8') as f:
#     samples = csv.reader(f)
#     next(samples)
#     train = list(samples)

# model_lookup_train = []
# labels_train = []
# for ind, item in tqdm(enumerate(train),total=len(train)):   
#     # 检索 rxn 的 Emb 
#     if item[0] in rxns.keys():
#         rxn = rxns[item[0]] 
#         model_lookup_train.append(rxn)
#         label_list = item[1].split('.')[:2]
#         label = '.'.join(label_list)
#         labels_train.append(label)
#     else:
#         # print(f'找不到rxn{ind}:{samples[ind]}')
#         continue

# print(len(model_lookup_train))  # 166918


# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC2_2024319_nolabel/model_lookup_train.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(np.array(model_lookup_train), file)

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC2_2024319_nolabel/labels_train.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(labels_train, file)



# ### test
# rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC2_2024319_nolabel/rxn_ec_emb.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)

# samples_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC2_2024319_nolabel/test_augmented.csv'
# with open(samples_file, 'r', encoding='utf-8') as f:
#     samples = csv.reader(f)
#     next(samples)
#     test = list(samples)

# model_lookup_test = []
# labels_test = []
# for ind, item in tqdm(enumerate(test),total=len(test)):   
#     # 检索 rxn 的 Emb 
#     if item[0] in rxns.keys():
#         rxn = rxns[item[0]] 
#         model_lookup_test.append(rxn)
#         label_list = item[1].split('.')[:2]
#         label = '.'.join(label_list)
#         labels_test.append(label)
#     else:
#         # print(f'找不到rxn{ind}:{test[ind]}')
#         continue

# print(len(model_lookup_test))    # 18816

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC2_2024319_nolabel/model_lookup_test.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(np.array(model_lookup_test), file)

# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC2_2024319_nolabel/labels_test.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(labels_test, file)


# #############################################
# #####  数据集：只取EC的前两位作为标签  制作 esm_emb [tensor格式] [字典格式]   
# #############################################
# ### train
# rxn_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC2_2024319_nolabel/rxn_ec_emb.pkl'
# with open(rxn_file,'rb') as f:
#     rxns = pickle.load(f)

# samples_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC2_2024319_nolabel/train_augmented.csv'
# with open(samples_file, 'r', encoding='utf-8') as f:
#     samples = csv.reader(f)
#     next(samples)
#     samples = list(samples)

# model_lookup_train = {}
# for ind, item in tqdm(enumerate(samples),total=len(samples)):    
#     # 检索 rxn 的 Emb 
#     if item[0] in rxns.keys():
#         rxn = rxns[item[0]] 
#         rxn = torch.Tensor(rxn)
#     else:
#         # print(f'找不到rxn{ind}:{samples[ind]}')
#         continue

    
#     label_list = item[1].split('.')[:2]
#     label = '.'.join(label_list)

#     if label in model_lookup_train.keys():
#         model_lookup_train[label].append(rxn)
#     else:
#         temp_list = []
#         temp_list.append(rxn)
        
#         model_lookup_train[label] = temp_list


# print(len(model_lookup_train))  # unique_ec: 63


# output_file = '/root/CLEAN/app/data/inputs/pred_rxn_EC2_2024319_nolabel/esm_emb_dict.pkl'
# with open(output_file, 'wb') as file:
#     pickle.dump(model_lookup_train, file)

