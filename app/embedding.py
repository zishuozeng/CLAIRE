from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)
import pickle

model, tokenizer = get_default_model_and_tokenizer()

rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)

example_rxn = ["C/C(C=O)=C\CC/C(C)=C/C=O.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.[H+].O>>COC(=O)CCCCCCCC=O", "CC(C=O)c1ccccc1.O.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)([O-])OP(=O)([O-])OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1>>CC(C(=O)O)c1ccccc1"]

fp = rxnfp_generator.convert_batch(example_rxn)
print(len(fp))

esm_emb_dict = {}
for ind, rxn in enumerate(example_rxn):
    esm_emb_dict[rxn] = fp[ind]
path = './data/inputs/rxn_ec_emb.pkl'
with open(path, 'wb') as file:
    pickle.dump(esm_emb_dict, file)
print(len(esm_emb_dict))