# Phenotype-Guided Generative Model for High-Fidelity Cardiac MRI Synthesis: Advancing Pretraining and Clinical Applications
## Environment Set Up
Install required packages:
```
conda create -n cpgg python=3.8
conda activate cpgg
pip install -r requirements.txt
```

## Train Models
```
sh script_train_phenotype_vae.sh
sh script_vae3d_kl8.sh
sh script_train_cpgg.sh
```
## Generate CMR
```
sh script_sample.sh
```