# 🔐 Vulnerability Detection with CodeBERT

This repository contains code to reproduce the vulnerability detection task using [CodeBERT](https://arxiv.org/abs/2002.08155), adapted and extended from the [CodeXGLUE](https://github.com/microsoft/CodeXGLUE) benchmark, extended with **VulScope**.

---

## 1. 🧐 Task Definition

The task is **binary classification** of C/C++ source code snippets:

- `1` → insecure (vulnerable)
- `0` → secure (non-vulnerable)

It aims to detect vulnerabilities such as **resource leaks**, **use-after-free**, or **DoS attacks** from raw function-level source code.

---


## 2. 📦 Dataset Format

We support two formats:
- `.jsonl` (line-delimited JSON)
- `.csv`

Each entry must contain:
```json
{
  "func" or "func_before": "<source_code_string>",
  "target": 0 or 1,
  "idx": "optional unique ID (auto-generated if missing)"
}
```

### 📂 Example File Structure
```
./../../paired_primevul/original/train.jsonl
./../../paired_primevul/original/val.jsonl
./../../paired_primevul/original/test.jsonl
```

---

## 3. ⚙️ Environment Setup

We recommend using Conda for environment management:

```bash
conda create -n vuln-detect python=3.8
conda activate vuln-detect
pip install torch transformers scikit-learn tqdm
```

Optional (for TensorBoard or Google Drive support):
- `tensorboardX`
- `gdown`

---

## 4. 🚀 Running the Model

### ✅ PrimeVul (Default)
```bash
python run.py \
  --do_train \
  --do_eval \
  --do_test \
  --train_data_file ./../../paired_primevul/original/train.jsonl \
  --eval_data_file ./../../paired_primevul/original/val.jsonl \
  --test_data_file ./../../paired_primevul/original/test.jsonl \
  --output_dir ./saved_models/primevul \
  --model_type roberta \
  --model_name_or_path microsoft/codebert-base \
  --saved_model_name primevul_epoch_20.bin \
  --block_size 512 \
  --train_batch_size 32 \
  --eval_batch_size 64 \
  --epoch 5 \
  --learning_rate 2e-5 \
  --early_stopping_patience 3 \
  --evaluate_during_training
```

### ✅ With VulScope Enabled (Similarity-Based Refinement)
```bash
python run.py \
  --do_test \
  --test_data_file ./../../paired_primevul/original/test.jsonl \
  --output_dir ./saved_models/primevul \
  --model_type roberta \
  --model_name_or_path microsoft/codebert-base \
  --saved_model_name primevul_epoch_20.bin \
  --block_size 512 \
  --eval_batch_size 64 \
  --ws 90 \
  --ov 10 \
  --use_vulscope \
  --similarity_db_path database_range_10.pkl
```

---

## 5. 📊 Evaluation Metrics

We report:
- **Accuracy**
- **F1 Score**
- **Precision, Recall**
- **Matthews Correlation Coefficient (MCC)**
- **Intersection over Union (IoU)**
- **Confusion Matrix**: TP, FP, TN, FN

Results and predictions are saved to:
```
saved_models/predictions.txt
```

---

## 7. 📝 Citation

If you use this repository, please cite:

#### 📄 CodeBERT
```bibtex
@inproceedings{feng2020codebert,
  title={Codebert: A pre-trained model for programming and natural languages},
  author={Feng, Zhangyin and Guo, Daya and Tang, Duyu and Duan, Nan and Feng, Xiaocheng and Gong, Ming and Shou, Linjun and Qin, Bing and Liu, Ting and Jiang, Daxin and others},
  booktitle={EMNLP},
  year={2020}
}
```

#### 📄 VulScope (if used)
```bibtex
@inproceedings{zhou2019devign,
  title={Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks},
  author={Zhou, Yaqin and Liu, Shangqing and Siow, Jingkai and Du, Xiaoning and Liu, Yang},
  booktitle={NeurIPS},
  year={2019}
}
```

---

Feel free to open issues or reach out for questions or improvements!
