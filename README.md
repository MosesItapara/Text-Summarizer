# 📝 Text Summarizer - End-to-End NLP Pipeline

An end-to-end text summarization project built with HuggingFace Transformers, fine-tuning **Google Pegasus** on the **SAMSum** dialogue dataset. The project follows a modular ML pipeline architecture with a FastAPI serving layer.

---

## 📁 Project Structure

```
Text-Summarizer/
├── config/
│   └── config.yaml                  # Artifact paths, model & tokenizer names
├── src/
│   └── textsummarizer/
│       ├── components/
│       │   ├── data_ingestion.py    # Downloads & saves dataset from HuggingFace
│       │   ├── data_transformation.py # Tokenizes dataset, saves Arrow format
│       │   └── model_trainer.py     # Fine-tunes Pegasus via HuggingFace Trainer
│       ├── config/
│       │   └── configuration.py     # ConfigurationManager — reads YAML configs
│       ├── entity/
│       │   └── __init__.py          # Frozen dataclasses for each pipeline stage
│       ├── pipeline/
│       │   ├── stage_1_data_ingestion_pipeline.py
│       │   ├── stage_2_data_transformation_pipeline.py
│       │   ├── stage_3_model_trainer_pipeline.py
│       │   └── stage_4_model_evaluation_pipeline.py
│       ├── logging/
│       │   └── __init__.py          # Custom logger
│       └── utils/
│           └── common.py            # read_yaml, create_directories helpers
├── research/
│   ├── data_ingestion.ipynb         # Stage 1 experiment notebook
│   ├── data_transformation.ipynb    # Stage 2 experiment notebook
│   ├── model_trainer.ipynb          # Stage 3 experiment notebook
│   └── model_evaluation.ipynb       # Stage 4 experiment notebook
├── artifacts/                       # Auto-generated — gitignored
│   ├── data_ingestion/
│   │   ├── ingested_train/          # Arrow dataset (train split)
│   │   └── ingested_test/           # Arrow dataset (test split)
│   ├── data_transformation/
│   │   ├── transformed_train/       # Tokenized train split
│   │   └── transformed_test/        # Tokenized test split
│   ├── model_trainer/
│   │   ├── pegasus-samsum-model/    # Fine-tuned model weights
│   │   └── tokenizer/               # Saved tokenizer
│   └── model_evaluation/
│       └── rouge_scores.csv         # ROUGE-1/2/L/Lsum scores
├── params.yaml                      # Hyperparameters & column names
├── main.py                          # Orchestrates all pipeline stages
├── app.py                           # FastAPI inference server
├── Dockerfile                       # Container definition
├── requirements.txt
└── setup.py
```

---

## 🔧 Tech Stack

| Layer | Tools |
|---|---|
| Model | Google Pegasus (`google/pegasus-xsum`) |
| Dataset | SAMSum (`knkarthick/samsum`) via HuggingFace Hub |
| Training | HuggingFace `transformers`, `Trainer` API |
| Data | HuggingFace `datasets`, Arrow format |
| Evaluation | `evaluate` library — ROUGE metrics |
| Serving | FastAPI + Uvicorn |
| Config | YAML + `python-box` ConfigBox |
| Containerization | Docker |
| Experiment Notebooks | Jupyter via VSCode |

---

## ⚙️ Configuration

### `config/config.yaml`
Controls artifact paths, dataset source, and model/tokenizer identifiers:

```yaml
artifacts_root: artifacts

artifacts:
  data_ingestion:
    root_dir: artifacts/data_ingestion
    ingested_train_dir: artifacts/data_ingestion/ingested_train
    ingested_test_dir: artifacts/data_ingestion/ingested_test
    dataset_name: "knkarthick/samsum"

  data_transformation:
    root_dir: artifacts/data_transformation
    tokenizer_name: "google/pegasus-xsum"

  model_trainer:
    root_dir: artifacts/model_trainer

  model_evaluation:
    root_dir: artifacts/model_evaluation
    metric_file_name: rouge_scores.csv
```

### `params.yaml`
Controls hyperparameters and dataset column mappings:

```yaml
data_transformation:
  input_column: dialogue
  target_column: summary
  max_input_length: 512
  max_target_length: 128

model_trainer:
  model_name: "google/pegasus-xsum"
  num_train_epochs: 1
  warmup_steps: 500
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  weight_decay: 0.01
  logging_steps: 10
  evaluation_strategy: "steps"
  eval_steps: 500
  save_steps: 1000
  gradient_accumulation_steps: 16

model_evaluation:
  input_column: dialogue
  target_column: summary
  batch_size: 2
  max_generate_length: 128
  num_beams: 8
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/MosesItapara/Text-Summarizer.git
cd Text-Summarizer
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline

```bash
python main.py
```

This runs all four stages in sequence:

```
Stage 1 — Data Ingestion       → downloads SAMSum, saves Arrow splits
Stage 2 — Data Transformation  → tokenizes with Pegasus tokenizer
Stage 3 — Model Trainer        → fine-tunes Pegasus, saves model + tokenizer
Stage 4 — Model Evaluation     → computes ROUGE scores, saves CSV
```

### 5. Start the API server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8080
```

Then POST to the `/summarize` endpoint:

```bash
curl -X POST "http://localhost:8080/summarize" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hannah: Hey, are you free tonight? Amanda: Yes! What do you have in mind? Hannah: Let'\''s grab dinner at that new Italian place."}'
```

Response:
```json
{
  "summary": "Hannah and Amanda are planning to have dinner at a new Italian restaurant tonight."
}
```

---

## 🔄 Pipeline Stages

### Stage 1 — Data Ingestion
- Loads the SAMSum dialogue summarization dataset from HuggingFace Hub (`knkarthick/samsum`)
- SAMSum contains 16,369 messenger-style conversations with human-written summaries
- Saves train and test splits to disk in Arrow format for fast downstream loading

### Stage 2 — Data Transformation
- Loads the Pegasus tokenizer (`google/pegasus-xsum`)
- Tokenizes `dialogue` (input) and `summary` (target) columns
- Applies truncation and padding to fixed sequence lengths
- Saves tokenized datasets in Arrow format with `input_ids`, `attention_mask`, and `labels`

### Stage 3 — Model Trainer
- Loads pre-trained `google/pegasus-xsum` from HuggingFace Hub
- Fine-tunes using HuggingFace `Trainer` with `DataCollatorForSeq2Seq`
- Supports mixed precision (`fp16`) automatically when GPU is available
- Saves the fine-tuned model and tokenizer to `artifacts/model_trainer/`

### Stage 4 — Model Evaluation
- Loads the fine-tuned model and runs inference on the test split in batches
- Computes ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum scores
- Saves results to `artifacts/model_evaluation/rouge_scores.csv`

---

## 🐳 Docker

### Build the image

```bash
docker build -t text-summarizer .
```

### Run the container

```bash
docker run -p 8080:8080 text-summarizer
```

---

## 📊 Evaluation Results

After fine-tuning, ROUGE scores on the SAMSum test set:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|---|---|---|---|---|
| Pegasus (fine-tuned) | ~0.42 | ~0.20 | ~0.34 | ~0.34 |

> Scores vary based on number of training epochs and hardware. Results above are indicative for 1 epoch on a single GPU.

---

## 🧪 Research Notebooks

Experiment notebooks in `research/` mirror each pipeline stage and are designed to be run independently for prototyping:

| Notebook | Purpose |
|---|---|
| `data_ingestion.ipynb` | Load and inspect the SAMSum dataset |
| `data_transformation.ipynb` | Experiment with tokenization strategies |
| `model_trainer.ipynb` | Prototype training loop and arguments |
| `model_evaluation.ipynb` | Compute ROUGE, visualise scores, test inference |

> **Note:** All notebooks must be run from the project root. The first cell sets `os.chdir(r"C:\Users\HP\Text-Summarizer")` to ensure correct path resolution.

---

## 🌐 API Reference

### `POST /summarize`

Summarizes a given input text using the fine-tuned Pegasus model.

**Request body:**
```json
{
  "text": "Your dialogue or article text here..."
}
```

**Response:**
```json
{
  "summary": "Generated summary text."
}
```

---

## 🛠️ Common Issues & Fixes

| Error | Cause | Fix |
|---|---|---|
| `BoxKeyError: tokenizer_dir` | Stale kernel cache | Restart kernel, re-run all cells |
| `FileNotFoundError: config\config.yaml` | Wrong working directory | Add `os.chdir(r"C:\Users\HP\Text-Summarizer")` as first cell |
| `DatasetNotFoundError: Samsung/samsum` | Dataset removed from Hub | Use `knkarthick/samsum` instead |
| `PermissionError: ingested_train` | Mixed `save_to_disk` + `to_csv` on same path | Use `save_to_disk` consistently; paths are folders not files |
| `AcceleratorError: CUDA device-side assert` | Out-of-range token IDs or bad inputs | Set `CUDA_LAUNCH_BLOCKING=1`, check vocab size vs max token ID |
| `TypeError: list[Path] in isinstance()` | Python < 3.10 with `ensure_annotations` | Use `List[Path]` from `typing` instead |

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)
- [SAMSum Dataset](https://huggingface.co/datasets/knkarthick/samsum) — Gliwa et al., 2019
- [Google Pegasus](https://huggingface.co/google/pegasus-xsum) — Zhang et al., 2020
