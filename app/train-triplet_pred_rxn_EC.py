import torch
import time
import os
import pickle
from src.CLAIRE.utils import * 
import torch.nn as nn
import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('-e', '--epoch', type=int, default=2000)
    parser.add_argument('-n', '--model_name', type=str, default='triplet')
    parser.add_argument('-d', '--hidden_dim', type=int, default=512)
    parser.add_argument('-o', '--out_dim', type=int, default=128)
    parser.add_argument('--adaptive_rate', type=int, default=200)
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()
    return args

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
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)
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

def mine_negative_random(anchor_ec, esm_emb):
    result_ec = random.choices(list(esm_emb.keys()))[0]

    while result_ec == anchor_ec:
        result_ec = random.choices(list(esm_emb.keys()))[0]
    
    return result_ec


class Triplet_dataset_with_mine_EC(torch.utils.data.Dataset):

    def __init__(self, esm_emb, labels):
        self.full_list = labels
        self.esm_emb = esm_emb
        
    def __len__(self):
        return len(self.full_list)
        # return 100

    def __getitem__(self, index):
        anchor_ec = self.full_list[index]  # EC_number
        anchor = random.choice(self.esm_emb[anchor_ec]).unsqueeze(0)    # emb

        pos = random.choice(self.esm_emb[anchor_ec]).unsqueeze(0)     # emb
        neg_ec = mine_negative_random(anchor_ec, self.esm_emb)
        neg = random.choice(self.esm_emb[neg_ec]).unsqueeze(0) 
        return anchor, pos, neg

def get_dataloader(esm_emb, labels, args):
    params = {
        'batch_size': 6000,
        'shuffle': True,
    }
    train_data = Triplet_dataset_with_mine_EC( esm_emb, labels)
    train_loader = torch.utils.data.DataLoader(train_data, **params)
    return train_loader


def train(model, args, epoch, train_loader,
          optimizer, device, dtype, criterion):
    model.train()
    total_loss = 0.
    start_time = time.time()

    for batch, data in enumerate(train_loader):
        optimizer.zero_grad()
        anchor, positive, negative = data
        anchor_out = model(anchor.to(device=device, dtype=dtype))
        positive_out = model(positive.to(device=device, dtype=dtype))
        negative_out = model(negative.to(device=device, dtype=dtype))

        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if args.verbose:
            lr = args.learning_rate
            ms_per_batch = (time.time() - start_time) * 1000
            cur_loss = total_loss 
            print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_loader):5d} batches | '
                  f'lr {lr:02.4f} | ms/batch {ms_per_batch:6.4f} | '
                  f'loss {cur_loss:5.2f}')
            start_time = time.time()
    # record running average training loss
    return total_loss/(batch + 1)


def main():
    seed_everything()
    ensure_dirs('./data/model')
    args = parse()
    torch.backends.cudnn.benchmark = True
    #======================== override args ====================#
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    lr, epochs = args.learning_rate, args.epoch
    model_name = args.model_name
    print('==> device used:', device, '| dtype used: ',
          dtype, "\n==> args:", args)
    #======================== ESM embedding  ===================#
    # loading ESM embedding
    esm_emb = pickle.load( 
        open('./data/inputs/pred_rxn_EC12/esm_emb_dict.pkl', 'rb'))
    # esm_emb = torch.tensor(esm_emb).to(device=device, dtype=dtype)
    labels = pickle.load(open('./data/inputs/pred_rxn_EC12/labels_train.pkl', 'rb'))
    print('### load.......done')
    #======================== initialize model =================#
    model = LayerNormNet(args.hidden_dim, args.out_dim, device, dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.TripletMarginLoss(margin=1, reduction='mean')
    best_loss = float('inf')
    train_loader = get_dataloader( esm_emb, labels, args)
    print("The number of unique EC numbers: ", len(esm_emb.keys()))
    #======================== training =======-=================#
    # training
    train_loss_plot = []
    for epoch in range(1, epochs + 1):
        temp_loss = []
        if epoch % args.adaptive_rate == 0 and epoch != epochs + 1:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, betas=(0.9, 0.999))
            # save updated model
            torch.save(model.state_dict(), './data/model/pred_rxn_EC12/random_10-4_' +
                       model_name + '_' + str(epoch) + '.pth')
            # delete last model checkpoint
            if epoch != args.adaptive_rate:
                os.remove('./data/model/pred_rxn_EC12/random_10-4_' + model_name + '_' +
                          str(epoch-args.adaptive_rate) + '.pth')
            # sample new distance map
            train_loader = get_dataloader(esm_emb, labels, args)
        # -------------------------------------------------------------------- #
        epoch_start_time = time.time()
        train_loss = train(model, args, epoch, train_loader,
                           optimizer, device, dtype, criterion)
        # only save the current best model near the end of training
        if (train_loss < best_loss and epoch > 0.8*epochs) :
            torch.save(model.state_dict(), './data/model/pred_rxn_EC12/random_10-4_' + model_name +'_best_'+ str(epoch) + '.pth')
            best_loss = train_loss
            print(f'Best from epoch : {epoch:3d}; loss: {train_loss:6.8f}')

        elapsed = time.time() - epoch_start_time
        print('-' * 75)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'training loss {train_loss:6.8f}')
        print('-' * 75)

        temp_loss.append(epoch)
        temp_loss.append(train_loss)
        train_loss_plot.append(temp_loss)

    # save final weights
    torch.save(model.state_dict(), './data/model/pred_rxn_EC12/random_10-4_' + model_name + str(epoch) +'_final'+ '.pth')
    with open('./results/inputs/pred_rxn_EC12/random_train_loss_10-4.pkl','wb') as file:
        pickle.dump(train_loss_plot, file)


if __name__ == '__main__':
    main()
