# CLAIRE
Contrastive Learning-based AnnotatIon for Reaction's Ec number

## 1.安装环境
```
cd CLAIRE/
conda create -n CLearning python==3.10
conda activate CLearning
pip install -r requirements.txt
```

安装`torch`：您可以安装 CPU 版本或者 GPU 版本的 torch

```
conda install pytorch==1.11.0 cpuonly -c pytorch (CPU)
conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch (GPU)
```
## 2.下载数据
您可以下载数据 [*inputs*]() ，放入`CLAIRE/dev/data`目录中，data 中包含`train_augmented.csv`、`test_augmented.csv`、`embedding`和 `yeast`。

```
train_augmented.csv 和 test_augmented.csv 为原始数据，model_lookup_train.pkl 和 model_lookup_test.pkl 为 rxn 的 embedding。
embedding 文件夹为对 rxn 进行 embedding 的详细步骤，该部分内容将在 part 3 中详细介绍。
pred_rxn_ECxxx 为对 ec 的三级划分，包含 labels 和 esm_emb ：{‘EC1’:[tensor1, tensor2, ... ,], ’EC2’:[tensor1, tensor2, ... ] ... .....}
yeast 文件夹中包含原始数据以及 positive 和 negative 的 embedding。
```
## 3.Embedding
如果你想对自己的数据进行编码，可以参考`data/embedding/embedding.py` 文件，embedding 由 rxnfp_embedding 和 drfp_embedding 两部分组成。

### 3.1 rxnfp embedding

运行 `embedding.py`获取rxnfp的 `rxn_ec_emb`，embedding的维度为256。
`example_rxn` 可替换为你的 rxn 列表。
生成的 embeddings 保存至 `CLAIRE/dev/data/embedding/rxnfp`文件夹下。

**rxnfp 运行环境**
在命令行输入以下命令，安装 rxnfp 的运行环境
```
bash rxnfp_env.sh
```

**rxnfp 使用方式**

```python
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)
import pickle

model, tokenizer = get_default_model_and_tokenizer()
rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
example_rxn = ["C/C(C=O)=C\CC/C(C)=C/C=O.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.[H+].O>>COC(=O)CCCCCCCC=O", "CC(C=O)c1ccccc1.O.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1>>CC(C(=O)O)c1ccccc1"]
fp = rxnfp_generator.convert_batch(example_rxn)
print(len(fp))
print(fp[:5])
esm_emb_dict = {}
for ind, rxn in enumerate(example_rxn):
    esm_emb_dict[rxn] = fp[ind]
path = './rxnfp/model_lookup_train.pkl'
with open(path, 'wb') as file:
    pickle.dump(esm_emb_dict, file)
print(len(esm_emb_dict))
```

### 3.2 drfp embedding

`embedding.py` 文件中详细说明了的 drfp_embedding 的生成方式。
生成的 embeddings 保存至 `CLAIRE/dev/data/embedding/drfp`文件夹下。

```python
### drfp embedding

## 安装 drfp 环境
# pip install drfp

# 命令行运行命令：drfp my_rxn_smiles.txt my_rxn_fps.pkl -d 256
# 其中，将my_rxn_smiles.txt替换为自己的rxn序列文件地址，my_rxn_fps.pkl 替换为保存的embedding的地址。
# -d 为生成的embedding维度
```

## 4.数据处理
`data_processing.py` 文件将 rxnfp 和 drfp 进行拼接，维度为512。
运行 `CLAIRE/dev/data/embedding/data_processing.py` 构建`model_lookup.pkl` 、` label.pkl`和`emb_eam.pkl`。

你可以根据预测EC的一位( EC:1 )，两位( EC:1.2 )或三位( EC:1.2.1 )来生成 labels 和 emb_esm。

**model_lookup**
```
model_lookup.shape 	# [len(rxn_list), 512]
Type(model_lookup)  # numpy
```
**label**
```
Len(label) 	# len(rxn_list)
Type(label)	# list
```
**esm_emb**
```
Len(esm_emb)	# len(unique_EC)
Type(esm_emb)	# dict
{‘EC1’:[tensor1, tensor2, ... ,], ’EC2’:[tensor1, tensor2, ... ] ... .....}
```

## 5.训练自己的数据

运行训练文件

```
python CLAIRE/dev/training/train-triplet_pred_rxn_EC.py
```

训练数据集的构建采用随机采样的方式：构建 esm_emb 字典` {‘EC1’:[tensor1, tensor2, ... ,], ’EC2’:[tensor1, tensor2, ... ] ... .....}`,并从样本集中随机选择一个 embedding 作为`anchor`，从 esm_emb 中与 anchor 相同的 EC 里随机抽样一个 embedding 作为 `positive`，从与 anchor 的不同的EC里随机抽样一个 embedding 作为 `negative`，从而构建一个三元组 `<anchor, positive, negative>`。其中，样本集中的每个样本都会作为 `anchor` 被选择。 我们训练一个三层的神经网络， 使 anchor 和 positive 之间的距离最小化， 和 negative 之间的距离最大化。

我们使用的损失函数是 `TripletMarginLoss`
设置 epoch 为 2000， batchsize 为 6000

训练好的模型保存至：`CLAIRE/dev/results/model`

## 6.inference
运行 `CLAIRE/dev/prediction/inference_EC.py`

导入`CLAIRE/dev/data/embedding/data_processing.py` 处理好保存的训练和测试数据。

**Inference：**
预训练模型位于：`CLAIRE/dev/results/model`
你可以根据预测EC的一位( EC:1 )，两位( EC:1.2 )或三位( EC:1.2.1 )来选择预训练模型。
同理训练数据也相应导入 EC 为一位、两位或三位的 labels。
导入训练数据(`model_lookup_train.pkl`)，与测试数据计算距离矩阵
 report_metrics = true 表示对预测结果进行评估。
 
```python
# EC calling results using maximum separation
infer_maxsep(train_data, test_data, train_labels, test_tags, test_labels, pretrained_model, report_metrics=True, gmm = "../results/test")
```

在评估阶段，我们使用GMM（高斯混合模型）测试查询化学式与预测的EC number之间的距离是否显著小于随机抽样化学式与随机抽样EC number之间的距离。测试的显著性越高，我们对模型预测的EC number的置信度就越高。 预测结果存储在 `CLAIRE/dev/results/test_prediction.csv` 中。

输出示例：
```
rxn_0,EC:4.1.1/0.9918,EC:1.10.1/0.2199,EC:1.1.1/0.0010
rxn_1,EC:2.3.1/0.9963,EC:4.2.1/0.0025,EC:3.1.2/0.0017
rxn_2,EC:3.5.4/0.9832,EC:2.4.1/0.0119,EC:1.8.1/0.0080
rxn_3,EC:2.3.1/0.8270,EC:6.2.1/0.2903,EC:4.2.1/0.0355
```
其中第一列（rxn_0）为 rxn 编号，代表的 rxn 序列， 第二列 （EC:4.1.1/0.9918） 是预测的 EC number 4.1.1 和 rxn_0 的实际的 EC number 相同的置信度。
