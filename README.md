# PruningQANN

## Installation

```bash
conda create -n pruningqann python=3.10 -y
conda activate pruningqann

pip install --upgrade pip
pip install -e .
```
For the LLaMA family, please install a compatible Transformers version:
```bash
pip install transformers==4.31.0
```

## Usage

For Baselines:

```bash
bash run_<baseline_name>.sh
```
The internal configuration of the `run_<baseline_name>.sh` script is as follows.

* GPU ID
* Sparsity ratio, choose from [0.5, 0.6, 0.7]
* Algorithm name (e.g., magnitude)
* Model path, e.g. meta-llama/Llama-2-7b-hf, facebook/opt-6.7b, etc.
* wbit (set to 4)
* Sparsity pattern/method, choose from unstructured, 2:4, or 4:8

Example:
```bash
bash run_llama2.sh 0 0.5 magnitude meta-llama/Llama-2-7b-hf 4 2:4
```

For Ours:
Set the algorithm name to `ours`.

Example:
```bash
bash run_llama1.sh 0 0.5 ours facebook/opt-6.7b 4 unstructured
```
