#!/bin/bash

# 创建 conda 环境
conda create -n rxnfp python=3.6 -y

# 激活 conda 环境
source activate rxnfp

# 安装必要的包
conda install -c rdkit rdkit=2020.03.3 -y
conda install -c tmap tmap -y
pip install rxnfp
