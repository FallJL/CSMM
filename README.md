

# CSMM

This repo will provide the code for reproducing the experiments in **Enhancing Code Search Fine-Tuning with
Momentum Contrastive Learning and
Cross-Modal Matching**.

CSMM is a fine-tuning method for code search that leverages momentum contrastive learning and
cross-modal matching.

## 1. Dependency

Our experiments are conducted on a 32GB NVIDIA Tesla V100 GPU with CUDA version 11.0.

```bash
conda create --name <your-env-name> python=3.9
conda activate <your-env-name>
pip install -r requirements.txt
```

## 2. Data Download

### 1) AdvTest dataset

```bash
cd data-download
bash advtest_download.sh
```

### 2) CosQA dataset

```bash
cd data-download
bash cosqa_download.sh
```

### 3) CSN dataset

```bash
cd data-download
bash csn_download.sh
```

## 3. Fine-Tuning

### 1) AdvTest dataset

```bash
bash run_advtest.sh
bash test_advtest.sh
```

### 2) CosQA dataset

The test procedure is included in the sh execution.

```bash
bash run_cosqa.sh
```

### 3) CSN dataset

The test procedure is included in the sh execution.

If you need to change the programming language, please set **lang** in the run_csn.sh file.

```bash
bash run_csn.sh
```
