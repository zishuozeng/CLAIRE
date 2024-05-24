import pandas as pd
import random
import pickle
import numpy as np

def extract_minor_ec(DF, N):
    ec_counts = DF['ec_label'].value_counts()
    ec_counts = pd.DataFrame({'ec': ec_counts.index, 'occurrence': ec_counts.values})
    ec_counts = ec_counts.sort_values(by='occurrence', ascending=False).reset_index(drop=True)
    minor_ec = list(ec_counts[ec_counts['occurrence'] < N].ec)
    return minor_ec

ecreact = pd.read_csv('ecreact-1.0.csv')

ec = list(ecreact.ec)
ec2 = []
ec3 = []
ec4 = []
for i in ec:
    splits = i.split('.')
    if splits[3] == '-':
        ec4.append('-')
    else:
        ec4.append(i)
    ###
    if splits[2] == '-':
        ec3.append('-')
    else:
        ec3.append('.'.join(splits[:3]))
    ###
    if splits[1] == '-':
        ec2.append('-')
    else:
        ec2.append('.'.join(splits[:2]))

ecreact['ec2'] = ec2
ecreact['ec3'] = ec3
ecreact['ec4'] = ec4 #[62222 rows x 6 columns]

ec4_counts = ecreact['ec4'].value_counts()
ec4_counts = pd.DataFrame({'ec': ec4_counts.index, 'occurrence': ec4_counts.values})
ec4_counts = ec4_counts.sort_values(by='occurrence', ascending=False).reset_index(drop=True) #[6088 rows x 2 columns]
minor_ec4 = list(ec4_counts[ec4_counts['occurrence'] < 10].ec)
len(minor_ec4) #5440

ec3_counts = ecreact['ec3'].value_counts()
ec3_counts = pd.DataFrame({'ec': ec3_counts.index, 'occurrence': ec3_counts.values})
ec3_counts = ec3_counts.sort_values(by='occurrence', ascending=False).reset_index(drop=True) #[277 rows x 2 columns]
minor_ec3 = list(ec3_counts[ec3_counts['occurrence'] < 10].ec)
len(minor_ec3) #101
ec3_counts.to_csv('ec3_counts.csv', index=False)

ec2_counts = ecreact['ec2'].value_counts()
ec2_counts = pd.DataFrame({'ec': ec2_counts.index, 'occurrence': ec2_counts.values})
ec2_counts = ec2_counts.sort_values(by='occurrence', ascending=False).reset_index(drop=True) #[73 rows x 2 columns]
minor_ec2 = list(ec2_counts[ec2_counts['occurrence'] < 10].ec)
len(minor_ec2) #7

#remove three-level EC numbers that have less than 10 occurrences
ecreact_filtered = ecreact[~ecreact['ec3'].isin(minor_ec3)] #[61817 rows x 6 columns]

###remove EC numbers from smiles strings
rxn_smiles = list(ecreact_filtered.rxn_smiles)
rxn_smiles1 = []
for i in rxn_smiles:
    splits1 = i.split('|')
    splits2 = i.split('>>')
    new_rxn = splits1[0] + '>>' + splits2[1]
    rxn_smiles1.append(new_rxn)

ecreact_filtered['rxn_smiles'] = rxn_smiles1

ecreact_filtered.to_csv('ecreact_filtered.csv', index=False) #[61817 rows x 6 columns]

#########################
### data augmentation ###
#########################
import pandas as pd
import random
import pickle
from itertools import permutations
import numpy as np
from sklearn.model_selection import train_test_split

def create_N_unique_shuffles(L, N):
    shuffled_lists = []
    for _ in range(N):
        shuffled_copy = L.copy()  # Create a copy of L to avoid modifying the original list
        random.shuffle(shuffled_copy)  # Shuffle the copy
        shuffled_lists.append(shuffled_copy)
    return shuffled_lists

ecreact_filtered = pd.read_csv('ecreact_filtered.csv') #[61817 rows x 6 columns]
ecreact_filtered = ecreact_filtered.drop('source', axis = 1)

train, test = pd.DataFrame(), pd.DataFrame()
for i in ecreact_filtered['ec3'].unique():
    df_i = ecreact_filtered[ecreact_filtered['ec3'] == i]
    train_i, test_i = train_test_split(df_i, test_size=0.1, stratify=df_i['ec3'])
    train = pd.concat([train, train_i])
    test = pd.concat([test, test_i])

train.to_csv('train.csv', index=False) #[55553 rows x 5 columns]
test.to_csv('test.csv', index=False) #[6264 rows x 5 columns]


# the augmentation process aims to augment reactants & products, respectively, 
# to min{ max{possible permutations}, 10}

#@@@@ train set @@@@#
ec3 = list(train.ec3)
train_rxn_smiles = list(train.rxn_smiles)

rxn_smiles1 = []
ec3_new = []
c = 0
for i in range(len(train_rxn_smiles)): # 
    rxn_smiles = train_rxn_smiles[i]
    reactants = rxn_smiles.split('>>')[0].split('.')
    products = rxn_smiles.split('>>')[1].split('.')
    ###
    reactants = random.sample(reactants, len(reactants))
    products = random.sample(products, len(products))
    ###
    if len(reactants) >3:
        reactant_permutations = create_N_unique_shuffles(reactants, 10)
    else:
        reactant_permutations = list(permutations(reactants))[:10]
    #
    if len(products) >3:
        product_permutations = create_N_unique_shuffles(products, 10)
    else:
        product_permutations = list(permutations(products))[:10]
    ###
    # reactant_permutations = random.sample(reactant_permutations, len(reactant_permutations))
    # product_permutations = random.sample(product_permutations, len(product_permutations))
    for j in reactant_permutations:
        for k in product_permutations:
            new_reaction = '.'.join(j) + '>>' + '.'.join(k)
            rxn_smiles1.append(new_reaction)
            # ec_new.append(ec[i])
            # ec2_new.append(ec2[i])
            ec3_new.append(ec3[i])
            #ec4_new.append(ec4[i])
    c += 1
    if c % 1000 == 0:
        print(c)

train_augmented = pd.DataFrame({'rxn_smiles':rxn_smiles1, 'ec3':ec3_new}) #[166918 rows x 2 columns]
train_augmented.to_csv('train_augmented.csv', index = False)

del rxn_smiles1, ec3_new, train_rxn_smiles, ec3

#@@@@ test set @@@@#
ec3 = list(test.ec3)
test_rxn_smiles = list(test.rxn_smiles)

rxn_smiles1 = []
ec3_new = []
c = 0
for i in range(len(test_rxn_smiles)): # 
    rxn_smiles = test_rxn_smiles[i]
    reactants = rxn_smiles.split('>>')[0].split('.')
    products = rxn_smiles.split('>>')[1].split('.')
    ###
    reactants = random.sample(reactants, len(reactants))
    products = random.sample(products, len(products))
    ###
    if len(reactants) >3:
        reactant_permutations = create_N_unique_shuffles(reactants, 10)
    else:
        reactant_permutations = list(permutations(reactants))[:10]
    #
    if len(products) >3:
        product_permutations = create_N_unique_shuffles(products, 10)
    else:
        product_permutations = list(permutations(products))[:10]
    ###
    # reactant_permutations = random.sample(reactant_permutations, len(reactant_permutations))
    # product_permutations = random.sample(product_permutations, len(product_permutations))
    for j in reactant_permutations:
        for k in product_permutations:
            new_reaction = '.'.join(j) + '>>' + '.'.join(k)
            rxn_smiles1.append(new_reaction)
            # ec_new.append(ec[i])
            # ec2_new.append(ec2[i])
            ec3_new.append(ec3[i])
            #ec4_new.append(ec4[i])
    c += 1
    if c % 1000 == 0:
        print(c)

test_augmented = pd.DataFrame({'rxn_smiles':rxn_smiles1, 'ec3':ec3_new}) #[18816 rows x 2 columns]

test_augmented.to_csv('test_augmented.csv', index = False)




###############################
##### reaction embeddings #####
###############################
from rxnfp.transformer_fingerprints import RXNBERTFingerprintGenerator
from rxnfp.transformer_fingerprints import get_default_model_and_tokenizer
from rxnfp.transformer_fingerprints import generate_fingerprints
import pandas as pd
import numpy as np
import pickle

train_augmented = pd.read_csv('train_augmented.csv')
test_augmented = pd.read_csv('test_augmented.csv')
rxns = list(set(train_augmented.rxn_smiles)) + list(set(test_augmented.rxn_smiles)) #

model, tokenizer = get_default_model_and_tokenizer()
rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)

rxn_ec_emb = {}
for i in rxns:
    rxn_ec_emb[i] = rxnfp_generator.convert(i)

pickle.dump(rxn_ec_emb, open('rxn_ec_emb.pkl', 'wb'))







