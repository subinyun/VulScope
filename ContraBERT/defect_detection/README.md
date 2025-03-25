
# 🔍 Vulnerability Detection with ContraBERT_C + VulScope

## 📌 Task: Defect Detection

We follow the [Defect Detection](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection) setting from CodeXGLUE for a fair comparison.  
We evaluate **ContraBERT_C** on the Devign dataset with and without **VulScope**, a token-windowing mechanism to enhance vulnerability localization.

---

## 🧩 Model

We use the pretrained model **[ContraBERT_C](https://drive.google.com/drive/u/1/folders/1F-yIS-f84uJhOCzvGWdMaOeRdLsVWoxN)**.

Once downloaded, place the model under:

```
./saved_models/pretrain_models/ContraBERT_C/
```

---

## ⚙️ Fine-Tuning (with VulScope)

We apply **VulScope**, a line-level token windowing strategy, to generate enhanced contextual features.

```bash
Pretrain_dir=./saved_models/pretrain_models/
Model_type=ContraBERT_C
OUT_DIR=./saved_models/finetune_models/vulscope_vulnerability_detection/${Model_type}
mkdir -p ${OUT_DIR}

python vulnerability_detection_vulscope.py \
    --output_dir=${OUT_DIR} \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=${Pretrain_dir}/${Model_type} \
    --do_train \
    --train_data_file=./../../paired_primevul/range_10/train.jsonl \
    --eval_data_file=../../../paired_primevul/range_10/val.jsonl \
    --test_data_file=./../../paired_primevul/original/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --ws 100 \
    --ov 15 \
    --seed 123456 2>&1 | tee ${OUT_DIR}/train.log
```

---

## 📈 Evaluation & Inference (with VulScope)

```bash
python vulnerability_detection_vulscope.py \
    --output_dir=${OUT_DIR} \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=${Pretrain_dir}/${Model_type} \
    --do_eval \
    --do_test \
    --train_data_file=./../../paired_primevul/range_10/train.jsonl \
    --eval_data_file=../../../paired_primevul/range_10/val.jsonl \
    --test_data_file=./../../paired_primevul/original/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --ws 100 \
    --ov 15 \
    --seed 123456 2>&1 | tee ${OUT_DIR}/test.log
```

---

## ⚖️ Baseline Fine-Tuning (without VulScope)

For comparison with the baseline:

```bash
python vulnerability_detection.py \
    --output_dir=./saved_models/finetune_models/vanilla_vulnerability_detection/ContraBERT_C \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=./saved_models/pretrain_models/ContraBERT_C \
    --do_train \
    --train_data_file=./../../paired_primevul/original/train.jsonl \
    --eval_data_file=../../../paired_primevul/original/val.jsonl \
    --test_data_file=./../../paired_primevul/original/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
      --use_vulscope \
      --similarity_db_path database_range_10.pkl
    --seed 123456 2>&1 | tee ./saved_models/finetune_models/vanilla_vulnerability_detection/ContraBERT_C/test.log
```

---

## 📁 Dataset Format

- **Input format**: `.jsonl` (or `.csv`)
- Required fields:
  - `"func"` or `"func_before"`: Source code as string
  - `"target"`: Label (0 or 1)
  - `"idx"`: Optional. If missing, a unique ID is automatically assigned

Example `.jsonl` line:
```json
{
  "func_before": "void foo() { int x = 0; if (x == 0) {...} }",
  "target": 1,
  "idx": "abc123"
}
```

---

## 🧪 Environment

- Python ≥ 3.7
- PyTorch ≥ 1.10
- Transformers ≥ 4.12
- pandas, sklearn, tqdm

---

## 📬 Citation / Credit

- [CodeXGLUE: Defect Detection Task](https://github.com/microsoft/CodeXGLUE)
- [ContraBERT](https://github.com/RUCKBReasoning/ContraBERT)

---
