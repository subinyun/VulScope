# 🔐 Unified Vulnerability Detection with VulScope

This repository provides training and evaluation pipelines for vulnerability detection using:
- **CodeBERT**
- **ContraBERT_C**
- **GraphCodeBERT**
- **StagedVulBERT**

All models support the optional integration of **VulScope**, windowing mechanism designed to enhance vulnerability localization by focusing on semantically important regions of code.

---

## 📌 Task Definition

We tackle the **binary classification** of C/C++ functions:
- `1` → Vulnerable
- `0` → Benign

---

## 📂 Dataset Format

We support:
- `.jsonl` 
- `.csv` 

Each entry includes:
```json
{
  "func" or "func_before": "<C/C++ source code>",
  "target": 0 or 1,
  "idx": "optional"
}
```
> ⚠️ **Note:** Not all model scripts automatically support both formats.  
> You must provide the dataset format compatible with the specific model:

| Model            | Expected Format | 
|------------------|------------------|
| CodeBERT         | `.jsonl`         |
| GraphCodeBERT    | `.jsonl`         | 
| ContraBERT_C     | `.jsonl`         | 
| StagedVulBERT    | `.csv`           | 

If you use the wrong format (e.g., pass `.csv` to a model expecting `.jsonl`), it may cause parsing errors or incorrect behavior.

### 📁 Example Paths
```
./../../paired_primevul/original/train.jsonl
./resource/dataset/test.csv
```

---

## 💾 Pretrained Models

Download the following pretrained models before fine-tuning:

| Model           | Link                                                                                      | Place in                               |
|----------------|--------------------------------------------------------------------------------------------|----------------------------------------|
| CodeBERT        | [HuggingFace](https://huggingface.co/microsoft/codebert-base)                             | `--model_name_or_path microsoft/codebert-base` |
| GraphCodeBERT   | [HuggingFace](https://huggingface.co/microsoft/graphcodebert-base)                        | `--model_name_or_path microsoft/graphcodebert-base` |
| ContraBERT_C    | [Google Drive](https://drive.google.com/drive/u/1/folders/1F-yIS-f84uJhOCzvGWdMaOeRdLsVWoxN) | `./saved_models/pretrain_models/ContraBERT_C/` |
| StagedVulBERT   | [MSP Model](https://drive.google.com/file/d/1frZLAmB2F0z1LLEwjVmoAtqKlPMg13uR/view?usp=sharing) | `./resource/staged-models/` |

---

## 🔽 Download Code Similarity Verification (CSV) Model and Database

The pretrained model and databases for the **Code Similarity Verification (CSV)** component are available via the following link:

📁 [Download VulScope CSV Checkpoint](https://drive.google.com/drive/folders/179tMJyppz34UuOOLc13A_Y2fTWrcBAzD?usp=sharing)

After downloading, place all files into the following directory:

```
VulScope/checkpoint-best-map/
```

The expected structure should look like this:

```
VulScope/
├── checkpoint-best-map/
│   ├── model.bin
│   ├── paired_cvefixes_database.pkl
│   └── paired_cvefixes_database_range_10.pkl
├── ...
```

- `model.bin`: CSV model weights
- `paired_cvefixes_database.pkl`: Full database of paired CVE-fix samples (range 5)
- `paired_cvefixes_database_range_10.pkl`: Full database of paired CVE-fix samples (range 10)

---

## ⚙️ Fine-Tuning Commands

### ✅ CodeBERT (PrimeVul)
```bash
python run.py \
  --do_train --do_eval --do_test \
  --train_data_file ./../../paired_primevul/original/train.jsonl \
  --eval_data_file ./../../paired_primevul/original/val.jsonl \
  --test_data_file ./../../paired_primevul/original/test.jsonl \
  --output_dir ./saved_models/primevul \
  --model_type roberta \
  --model_name_or_path microsoft/codebert-base \
  --block_size 512 \
  --train_batch_size 32 --eval_batch_size 64 \
  --epoch 5 --learning_rate 2e-5 \
  --evaluate_during_training
```

### ✅ CodeBERT + VulScope
```bash
python run.py \
  ... \
  --use_vulscope \
  --ws 90 --ov 10 \
  --similarity_db_path database_range_10.pkl
```

---

### ✅ ContraBERT_C + VulScope
```bash
python vulnerability_detection_vulscope.py \
  --model_type roberta \
  --model_name_or_path ./saved_models/pretrain_models/ContraBERT_C \
  --tokenizer_name microsoft/codebert-base \
  --train_data_file ./../../paired_primevul/range_10/train.jsonl \
  --eval_data_file ./../../paired_primevul/range_10/val.jsonl \
  --test_data_file ./../../paired_primevul/original/test.jsonl \
  --output_dir ./saved_models/finetune_models/vulscope_vulnerability_detection/ContraBERT_C \
  --block_size 400 --ws 100 --ov 15 \
  --epoch 5 --train_batch_size 32 --eval_batch_size 64 \
  --learning_rate 2e-5 --max_grad_norm 1.0 \
  --evaluate_during_training
```

---

### ✅ GraphCodeBERT + VulScope
```bash
python vulnerability_detection_graphcodebert.py \
  --model_name_or_path microsoft/graphcodebert-base \
  --tokenizer_name microsoft/graphcodebert-base \
  --do_train --do_eval --do_test \
  --train_data_file ./../../paired_primevul/range_10/train.jsonl \
  --eval_data_file ./../../paired_primevul/range_10/val.jsonl \
  --test_data_file ./../../paired_primevul/original/test.jsonl \
  --output_dir ./saved_models/finetune_models/vulscope_vulnerability_detection/GraphCodeBERT \
  --code_length 512 --data_flow_length 128 \
  --train_batch_size 32 --eval_batch_size 64 \
  --learning_rate 2e-5 --epochs 5 \
  --max_grad_norm 1.0 --evaluate_during_training
```

---

### ✅ StagedVulBERT + VulScope (Big-Vul)
```bash
python Entry/StagedBert_vul.py \
  --do_train --do_eval --do_test \
  --train_data_file ./resource/dataset/train.csv \
  --eval_data_file ./resource/dataset/valid.csv \
  --test_data_file ./resource/dataset/test.csv \
  --output_dir ./saved_models/stagedvulbert \
  --block_size 512 --eval_batch_size 32 \
  --model_name=pretrain_model.bin
```

---

## 📈 Evaluation Metrics

We report:
- Accuracy
- F1 Score
- Precision / Recall
- Matthews Correlation Coefficient (MCC)
- Intersection over Union (IoU)
- Confusion Matrix (TP, FP, TN, FN)

Predictions saved at:
```
./saved_models/.../predictions.txt
```

---

## 📦 Environment Setup

Install requirements:

```bash
pip install torch transformers scikit-learn pandas tqdm
```

Optional:
```bash
pip install captum libclang tokenizers
```

---

## 📚 Citation

Please cite the following:

**CodeBERT**
```bibtex
@inproceedings{feng2020codebert,
  title={Codebert: A pre-trained model for programming and natural languages},
  author={Feng, Zhangyin et al.},
  booktitle={EMNLP}, year={2020}
}
```

**GraphCodeBERT**
```bibtex
@inproceedings{guo2020graphcodebert,
  title={GraphCodeBERT: Pre-training code representations with data flow},
  author={Guo, Daya et al.},
  booktitle={ICLR}, year={2021}
}
```
**ContraBERT**
```bibtex
@inproceedings{liu2023contrabert,
  title={Contrabert: Enhancing code pre-trained models via contrastive learning},
  author={Liu, Shangqing and Wu, Bozhi and Xie, Xiaofei and Meng, Guozhu and Liu, Yang},
  booktitle={2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE)},
  pages={2476--2487},
  year={2023},
  organization={IEEE}
}
```

**StagedVulBERT**
```bibtex
@article{jiang2024stagedvulbert,
  title={StagedVulBERT: Multi-Granular Vulnerability Detection with a Novel Pre-trained Code Model},
  author={Jiang, Yuan and Zhang, Yujian and Su, Xiaohong and Treude, Christoph and Wang, Tiantian},
  journal={IEEE Transactions on Software Engineering},
  year={2024},
  publisher={IEEE}
}
```

---

Feel free to open issues or PRs for improvements!