# DEMO
你可以通过运行 `CLAIRE/DEMO.py` 初步使用
**环境安装**
```
conda activate rxnfp
cd CLAIRE/
python DEMO.py
```
首先，导入训练数据
```python
### train data
# train_embedding 512-dim
train_file = './dev/data/model_lookup_train.pkl'
with open (train_file, 'rb') as file:
    train_data = pickle.load(file)

# train_labels
labels_file = './dev/data/pred_rxn_EC123/labels_train_ec3.pkl'
with open (labels_file, 'rb') as file:
    train_labels = pickle.load(file)
```

你可以使用我们的测试数据进行初步测试。
```python
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
```

如果你要预测一条或多条序列的EC，你可以使用`CLAIRE/dev/data/embedding/embedding.py`对你的序列进行编码
**rxnfp Embedding：**
测试一条：传入字符串
```python
example_rxn = "C/C(C=O)=C\CC/C(C)=C/C=O.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.[H+].O>>COC(=O)CCCCCCCC=O"
rxnfp = rxnfp_generator.convert(example_rxn)
```
测试多条：传入列表，每个 rxn 用“`，`”隔开。
```python
example_rxn = [" ", " ", " ", " ", " "]
rxnfp = rxnfp_generator.convert_batch(example_rxn)
```

**drfp embedding**

```python
### drfp embedding

## 安装 drfp 环境
# pip install drfp

# 命令行运行命令：drfp my_rxn_smiles.txt my_rxn_fps.pkl -d 256
# 其中，将my_rxn_smiles.txt替换为自己的rxn序列文件地址，my_rxn_fps.pkl 替换为保存的embedding的地址。
# -d 为生成的embedding维度
```


**concat rxnfp and drfp**

```python
test_data = []
for ind, item in enumerate(rxnfp):
    rxn_emb = np.concatenate((np.reshape(item, (1,256)), np.reshape(drfp[ind], (1,256))), axis=1)
    test_data.append(rxn_emb)
test_data = np.concatenate(test_data,axis=0)
```

**Inference：**
```python
# EC calling results using maximum separation
infer_maxsep(train_data, test_data, train_labels, test_tags, test_labels, pretrained_model,out_filename='./dev/results/demo', gmm = './dev/GMM/gmm_ensumble.pkl')
```
