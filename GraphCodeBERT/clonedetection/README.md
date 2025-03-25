# 🔐 Vulnerability Detection with GraphCodeBERT + VulScope

We evaluate **GraphCodeBERT** on the [PrimeVul dataset](https://github.com/SunHaozhe/PrimeVul) both **with and without VulScope**, a window-based enhancement for dataflow-guided token representation.

---

## 🧠 Model

We use the pretrained model **GraphCodeBERT** from HuggingFace:
```
microsoft/graphcodebert-base
```

---

## ⚙️ Fine-Tuning (with VulScope)

We apply **VulScope**, a DFG-aware token masking mechanism to enhance vulnerability localization.

```bash
Pretrain_dir=microsoft/graphcodebert-base
Model_type=GraphCodeBERT
OUT_DIR=./saved_models/finetune_models/vulscope_vulnerability_detection/${Model_type}
mkdir -p ${OUT_DIR}

python vulnerability_detection_graphcodebert.py \
    --output_dir=${OUT_DIR} \
    --config_name=${Pretrain_dir} \
    --model_name_or_path=${Pretrain_dir} \
    --tokenizer_name=${Pretrain_dir} \
    --do_train \
    --train_data_file=./../../paired_primevul/range_10/train.jsonl \
    --eval_data_file=../../../paired_primevul/range_10/val.jsonl \
    --test_data_file=./../../paired_primevul/original/test.jsonl \
    --epochs 5 \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee ${OUT_DIR}/train.log
```

---

## 🧪 Evaluation & Inference (with VulScope)

```bash
python vulnerability_detection_graphcodebert.py \
    --output_dir=${OUT_DIR} \
    --config_name=${Pretrain_dir} \
    --model_name_or_path=${Pretrain_dir} \
    --tokenizer_name=${Pretrain_dir} \
    --do_eval \
    --do_test \
    --train_data_file=./../../paired_primevul/range_10/train.jsonl \
    --eval_data_file=../../../paired_primevul/range_10/val.jsonl \
    --test_data_file=./../../paired_primevul/original/test.jsonl \
    --epochs 5 \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee ${OUT_DIR}/test.log
```

---

## ⚖️ Baseline Fine-Tuning (without VulScope)

```bash
python vulnerability_detection_graphcodebert.py \
    --output_dir=./saved_models/finetune_models/vanilla_vulnerability_detection/GraphCodeBERT \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --do_train \
    --train_data_file=./../../paired_primevul/original/train.jsonl \
    --eval_data_file=../../../paired_primevul/original/val.jsonl \
    --test_data_file=./../../paired_primevul/original/test.jsonl \
    --epochs 5 \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee ./saved_models/finetune_models/vanilla_vulnerability_detection/GraphCodeBERT/test.log
```

---

## 🧾 Dataset Format

- **Format**: `.jsonl` (line-by-line JSON)
- Required fields:
  - `func_before` or `func`: Source code
  - `target`: Label (0 or 1)
  - `idx`: Optional, auto-assigned if missing

```json
{
  "func_before": "void example() { int x = 0; if(x == 0) { ... } }",
  "target": 1
}
```

---

## 📦 Environment Requirements

- Python ≥ 3.7
- PyTorch ≥ 1.10
- Transformers ≥ 4.12
- tree_sitter
- pandas, sklearn, tqdm

---

## 📊 Result Reporting

Evaluation metrics:
- Accuracy
- F1 Score
- Precision / Recall
- MCC (Matthews Correlation Coefficient)
- IoU (Intersection over Union)

Prediction logs are saved in:
```
${OUT_DIR}/predictions.txt
```

---

## 🙌 Credits

- [CodeXGLUE Defect Detection](https://github.com/microsoft/CodeXGLUE)
- [GraphCodeBERT](https://github.com/microsoft/CodeBERT)

---

