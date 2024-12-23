# CLAIRE
Contrastive Learning-based AnnotatIon for Reaction's Ec number

## 1.Environment setup
Follow instructions in `CLAIRE/README.md` to set up environment for this application.


## 2.Data
Training and testing data are placed under `CLAIRE/dev/data`

## 3.Embedding
Reaction embedding can be done following the script `data/embedding/embedding.py`.

## 4.Training

Run the following script to train the model

```
python CLAIRE/dev/training/train-pred_rxn_EC.py
```

The trained model is saved at `CLAIRE/dev/results/model`

## 5.Inference
Run the script `CLAIRE/dev/prediction/inference_EC.py` to make predictions for reaction's EC number,

or follow `CLAIRE/README.md` to make prediction step by step.

The models can be found at `CLAIRE/dev/results/model`, where you can choose to predict the first level (EC:1.-.-.-), first two levels (e.g., EC 1.2.-.-), or the first three levels (e.g., EC 1.2.3.-) of the EC number. 

Prediction output is saved at `CLAIRE/dev/results/`.
