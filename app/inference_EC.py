# from src.CLEAN.infer import infer_maxsep
import torch
from src.CLEAN.utils import * 
from src.CLEAN.model import LayerNormNet
from src.CLEAN.distance_map import *
from src.CLEAN.evaluate import *
import pandas as pd
import torch.nn as nn

class LayerNormNet(nn.Module):
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(LayerNormNet, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype

        self.fc1 = nn.Linear(256, hidden_dim, dtype=dtype, device=device)
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=dtype, device=device)
        self.ln2 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x

def dist_map_helper(keys1, lookup1, keys2, lookup2):
    print("The embedding sizes for train and test:",
          lookup2.size(), lookup1.size())
    dist = {}
    for i, key1 in tqdm(enumerate(keys1), total=len(keys1)):
        current = lookup1[i].unsqueeze(0)
        dist_norm = (current - lookup2).norm(dim=1, p=2)
        dist_norm = dist_norm.detach().cpu().numpy()
        dist[key1] = {}
        for j, key2 in enumerate(keys2):
            dist[key1][key2] = dist_norm[j]
    return dist


def get_pred_labels(out_filename, pred_type="_maxsep"):
    file_name = out_filename+pred_type
    result = open(file_name+'.csv', 'r')
    csvreader = csv.reader(result, delimiter=',')
    pred_label = []
    for row in csvreader:
        ec_i = row[1].split(":")[1].split("/")[0]
        pred_label.append(ec_i)
    return pred_label

def write_max_sep_choices(df, csv_name, first_grad=False, use_max_grad=False, gmm = None):
    out_file = open(csv_name + '_maxsep.csv', 'w', newline='')
    csvwriter = csv.writer(out_file, delimiter=',')
    all_test_EC = set()
    for col in df.columns:  
        ec = []
        # 获取df某一列最小的十个元素
        smallest_10_dist_df = df[col].nsmallest(10)
        dist_lst = list(smallest_10_dist_df)    # [3.5313403259351617e-06, 0.052937183529138565, 0.06757994741201401, 0.1062488704919815, 0.10870343446731567, 0.13690374791622162, 0.18738529086112976, 0.1886766403913498, 0.1971900761127472, 0.19742491841316223]

        max_sep_i = maximum_separation(dist_lst, first_grad, use_max_grad)
        
        for i in range(max_sep_i+1):
            EC_i = smallest_10_dist_df.index[i]
            dist_i = dist_lst[i] 
            if gmm != None:
                gmm_lst = pickle.load(open(gmm, 'rb'))
                dist_i = infer_confidence_gmm(dist_i, gmm_lst)
            dist_str = "{:.4f}".format(dist_i)
            all_test_EC.add(EC_i)
            ec.append('EC:' + str(EC_i) + '/' + dist_str)
        ec.insert(0, col)
        csvwriter.writerow(ec)
    return

def infer_maxsep(train_data, test_data, train_tags, test_tags, test_labels, pretrained_model, report_metrics = True, 
                 pretrained=True, gmm = None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32

    # load checkpoints
    # NOTE: change this to LayerNormNet(512, 256, device, dtype) 
    # and rebuild with [python build.py install]
    # if inferencing on model trained with supconH loss
    model = LayerNormNet(512, 128, device, dtype)
    
    if pretrained:
        try:
            checkpoint = torch.load('./data/pretrained/'+ train_data +'.pth', map_location=device)
        except FileNotFoundError as error:
            raise Exception('No pretrained weights for this training data')
    else:
        try:
            checkpoint = torch.load(pretrained_model, map_location=device)
        except FileNotFoundError as error:
            raise Exception('No model found!')
            
    model.load_state_dict(checkpoint)
    model.eval()
    # load precomputed EC cluster center embeddings if possible
    if train_data == "split70":
        emb_train = torch.load('./data/pretrained/70.pt', map_location=device)
    elif train_data == "split100":
        emb_train = torch.load('./data/pretrained/100.pt', map_location=device)
    else:
        # 这里导入训练数据的 emb
        emb_train = model(torch.tensor(train_data).to(device=device, dtype=dtype))
    
    # 导入测试数据的 emb
    emb_test = model(torch.tensor(test_data).to(device=device, dtype=dtype))
   

    eval_dist = dist_map_helper(test_tags, emb_test, train_tags, emb_train)
    

    seed_everything()
    eval_df = pd.DataFrame.from_dict(eval_dist)

    ensure_dirs("./results")
    out_filename = "results/inputs/test"  # 预测结果保存地址

    write_max_sep_choices(eval_df, out_filename, gmm=gmm)

    #精度验证
    if report_metrics:
        pred_label = get_pred_labels(out_filename, pred_type='_maxsep')
        print(f'len(pred_label):{len(pred_label)}')
        true_label = test_labels
        print(f'len(true_label):{len(true_label)}')
        print('start_eval')
        
        f1 = f1_score(true_label, pred_label, average='weighted')
        acc = accuracy_score(true_label, pred_label)       
        print(f'f1:{f1}   |   acc:{acc}')

        

if __name__ == '__main__':

    ### train data
    train_file = './data/inputs/pred_rxn_EC12/model_lookup_train.pkl'
    with open (train_file, 'rb') as file:
        train_data = pickle.load(file)

    labels_file = './data/inputs/pred_rxn_EC12/labels_train.pkl'
    with open (labels_file, 'rb') as file:
        train_labels = pickle.load(file)


    ### test data
    test_file = './data/inputs/pred_rxn_EC12/model_lookup_test.pkl'
    # test_file = '/root/CLEAN/app/plot/embedding_1000.pkl'
    with open (test_file, 'rb') as file:
        test_data = pickle.load(file)

    labels_file = './data/inputs/pred_rxn_EC12/labels_test.pkl'
    # labels_file = '/root/CLEAN/app/plot/labels_1000.pkl'
    with open (labels_file, 'rb') as file:
        test_labels = pickle.load(file)

    # test_data = np.r_[test_data[:50], test_data[-50:]]
    test_tags = []
    for i in range(len(test_data)):
        test_tags.append('rxn_' + str(i))


    pretrained_model = './data/model/pred_rxn_EC12/random_10-4_triplet2000_final.pth'
    infer_maxsep(train_data, test_data, train_labels, test_tags,test_labels, pretrained_model, report_metrics=True, pretrained=False, gmm = './data/pretrained/gmm_ensumble.pkl')
    # infer_maxsep(train_data, test_data, labels, test_tags, report_metrics=False, pretrained=False, model_name='tmp_triplet', gmm = './data/pretrained/gmm_ensumble.pkl')





