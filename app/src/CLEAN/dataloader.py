import torch
import random
from .utils import format_esm


def mine_hard_negative(dist_map, knn=10):
    #print("The number of unique EC numbers: ", len(dist_map.keys()))
    ecs = list(dist_map.keys())
    negative = {}
    for i, target in enumerate(ecs):
        # 依 dist_map[target].items() 进行升序排序
        sort_orders = sorted(
            dist_map[target].items(), key=lambda x: x[1], reverse=False)        # [('1', 0.0), ('3', 0.8698208004930196), ('4', 1.3072911081939287), ('5', 1.5913367136837888), ('2', 1.5989677430503912), ('-127646', 20.648262552000357), ('-127644', 21.627502068324542), ('-127645', 22.912026538930046), ('-127643', 23.08076107281079), ('-127647', 25.570262520671182)]
        if sort_orders[1][1] != 0:
            # 创建一个名为 freq 的列表，其中包含了 sort_orders 中选定范围内元素的倒数。具体来说，对于每个选定的元素 (a, b)，1/b 被添加到 freq 列表中
            freq = [1/i[1] for i in sort_orders[1:1 + knn]]
            # 创建一个名为 neg_ecs 的列表，其中包含了 sort_orders 中选定范围内元素的第一个元素。具体来说，对于每个选定的元素 (a, b)，a 被添加到 neg_ecs 列表中
            neg_ecs = [i[0] for i in sort_orders[1:1 + knn]]
        elif sort_orders[2][1] != 0:
            freq = [1/i[1] for i in sort_orders[2:2+knn]]
            neg_ecs = [i[0] for i in sort_orders[2:2+knn]]
        elif sort_orders[3][1] != 0:
            freq = [1/i[1] for i in sort_orders[3:3+knn]]
            neg_ecs = [i[0] for i in sort_orders[3:3+knn]]
        else:
            freq = [1/i[1] for i in sort_orders[4:4+knn]]
            neg_ecs = [i[0] for i in sort_orders[4:4+knn]]

        # 归一化
        normalized_freq = [i/sum(freq) for i in freq]
        negative[target] = {
            'weights': normalized_freq,
            'negative': neg_ecs
        }
    return negative


def mine_negative(anchor, id_ec, ec_id, mine_neg):
    anchor_ec = id_ec[anchor]
    pos_ec = random.choice(anchor_ec)
    neg_ec = mine_neg[pos_ec]['negative']
    weights = mine_neg[pos_ec]['weights']
    result_ec = random.choices(neg_ec, weights=weights, k=1)[0]
    while result_ec in anchor_ec:
        result_ec = random.choices(neg_ec, weights=weights, k=1)[0]
    neg_id = random.choice(ec_id[result_ec])
    return neg_id


def random_positive(id, id_ec, ec_id):
    pos_ec = random.choice(id_ec[id])    # 随机获得 ec
    pos = id    # Uniprot
    if len(ec_id[pos_ec]) == 1:
        return pos + '_' + str(random.randint(0, 9))
    while pos == id:
        pos = random.choice(ec_id[pos_ec])
    return pos


class Triplet_dataset_with_mine_EC(torch.utils.data.Dataset):

    def __init__(self, esm_emb, labels):
        # self.id_ec = id_ec
        # self.ec_id = ec_id
        self.full_list = labels # ec_list 
        self.esm_emb = esm_emb
        # self.mine_neg = mine_neg
        # for ec in ec_id.keys():
        #     if '-' not in ec:
        #         self.full_list.append(ec)

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self):
        # anchor_label 为 0 或 1 , str 
        # anchor_label = self.full_list[index]

        # full_list 前面都为1，后面都为0 ，获取交界索引
        count_labels = self.full_list.count('1')

        # 随机选择一个anchor_emb
        anchor_id = random.randint(0, len(self.full_list)-1)
        anchor = torch.tensor(self.esm_emb[anchor_id] )
        if self.full_list[anchor_id] == '1':
            pos = torch.tensor(random.choice(self.esm_emb[:count_labels-1]))
            neg = torch.tensor(random.choice(self.esm_emb[count_labels:]))
        else:
            pos = torch.tensor(random.choice(self.esm_emb[count_labels:]))
            neg = torch.tensor(random.choice(self.esm_emb[:count_labels-1]))
        

        # a = torch.load('./data/esm_data/' + anchor + '.pt')     # {'label': 'W5EP13_3', 'mean_representations': {33: tensor([-0.1040,  0.3126,  0.1471,  ...,  0.0028, -0.0350, -0.0228])}}
        # p = torch.load('./data/esm_data/' + pos + '.pt')
        # n = torch.load('./data/esm_data/' + neg + '.pt')
        return anchor, pos, neg


class MultiPosNeg_dataset_with_mine_EC(torch.utils.data.Dataset):

    def __init__(self, id_ec, ec_id, mine_neg, n_pos, n_neg):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.full_list = []
        self.mine_neg = mine_neg
        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        anchor_ec = self.full_list[index]
        anchor = random.choice(self.ec_id[anchor_ec])
        a = format_esm(torch.load('./data/esm_data/' +
                       anchor + '.pt')).unsqueeze(0)
        data = [a]
        for _ in range(self.n_pos):
            pos = random_positive(anchor, self.id_ec, self.ec_id)
            p = format_esm(torch.load('./data/esm_data/' +
                           pos + '.pt')).unsqueeze(0)
            data.append(p)
        for _ in range(self.n_neg):
            neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)
            n = format_esm(torch.load('./data/esm_data/' +
                           neg + '.pt')).unsqueeze(0)
            data.append(n)
        return torch.cat(data)
