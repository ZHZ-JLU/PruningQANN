# PruningQANN

## Installation

```bash
conda create -n pruningqann python=3.10 -y
conda activate pruningqann

pip install --upgrade pip
pip install -e .
```

```bash
pip install transformers==4.31.0
```

## Usage

* GPU ID
* Sparsity ratio, choose from [0.5, 0.6, 0.7]
* Algorithm name (e.g., magnitude)
* Model path, e.g. meta-llama/Llama-2-7b-hf.
* wbit (set to 4)
* abit (set to 16)
* unstructured
* rho only for ours

For Baselines

Example:
```bash
bash run_llama2.sh 0 0.5 magnitude meta-llama/Llama-2-7b-hf 4 16 unstructured 1e-5
```

For Ours:
Set the algorithm name to `ours`.

Example:
```bash
bash run_llama2.sh 0 0.5 ours meta-llama/Llama-2-7b-hf 4 16 unstructured 1e-5
```
