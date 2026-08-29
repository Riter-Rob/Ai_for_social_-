"""
Ethiopia/Model/fine_tune_colab.py

Google Colab fine-tuning script for Ethiopian crisis detection.

Model: Davlan/afro-xlmr-base (pre-trained on 17 African languages incl. Amharic)
Task:  Binary classification: 1 = crisis/distress, 0 = safe

INSTRUCTIONS:
  1. Upload this file to Google Colab (colab.research.google.com)
  2. Upload Ethiopia/Dataset/seed_combined.csv to the Colab session
  3. Set Runtime → Change runtime type → GPU (T4)
  4. Run all cells
  5. Download the saved model folder 'ethiopian_crisis_model/'
  6. Place it in Ethiopia/Model/ethiopian_crisis_model/ in your project

NOTE: You can also run this locally if you have a GPU or accept slower CPU training.
"""

# ─── CELL 1: Install dependencies ────────────────────────────────────────────
# Uncomment in Colab:
# !pip install transformers datasets torch scikit-learn pandas -q

import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# ─── CELL 2: Configuration ────────────────────────────────────────────────────
MODEL_NAME = "Davlan/afro-xlmr-base"   # African multilingual model
MAX_LENGTH = 128                         # Max token length (social media posts are short)
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 2e-5
OUTPUT_DIR = "./ethiopian_crisis_model"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ─── CELL 3: Load dataset ─────────────────────────────────────────────────────
# In Colab, upload seed_combined.csv first then run this cell
df = pd.read_csv("seed_combined.csv", encoding="utf-8-sig")
print(f"Dataset loaded: {len(df)} rows")
print(f"Label distribution:\n{df['label'].value_counts()}")
print(f"\nLanguage distribution:\n{df['language'].value_counts()}")

# Remove any rows with empty text
df = df.dropna(subset=["text"])
df = df[df["text"].str.strip() != ""]
df = df.reset_index(drop=True)
print(f"\nAfter cleaning: {len(df)} rows")

# ─── CELL 4: Train/Validation split ──────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=SEED,
    stratify=df["label"],
)
print(f"Train: {len(X_train)} | Val: {len(X_val)}")

# ─── CELL 5: Tokenizer ───────────────────────────────────────────────────────
print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class EthiopianCrisisDataset(Dataset):
    """PyTorch dataset for Ethiopian crisis detection."""

    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


train_dataset = EthiopianCrisisDataset(X_train, y_train, tokenizer)
val_dataset   = EthiopianCrisisDataset(X_val,   y_val,   tokenizer)
print(f"Datasets ready. Train batches: {len(train_dataset)}")

# ─── CELL 6: Load model ───────────────────────────────────────────────────────
print(f"\nLoading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True,
)
model = model.to(device)
print("Model loaded successfully.")

total_params = sum(p.numel() for p in model.parameters())
trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,} | Trainable: {trainable:,}")

# ─── CELL 7: Training arguments ───────────────────────────────────────────────
def compute_metrics(eval_pred):
    """Custom metrics for HuggingFace Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, output_dict=True)
    return {
        "accuracy": acc,
        "f1_crisis":    report.get("1", {}).get("f1-score", 0),
        "f1_safe":      report.get("0", {}).get("f1-score", 0),
        "precision":    report.get("weighted avg", {}).get("precision", 0),
        "recall":       report.get("weighted avg", {}).get("recall", 0),
    }


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_crisis",
    greater_is_better=True,
    logging_steps=10,
    report_to="none",     # Disable wandb
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# ─── CELL 8: Train ────────────────────────────────────────────────────────────
print("\n=== Starting Training ===")
trainer.train()

# ─── CELL 9: Evaluate ─────────────────────────────────────────────────────────
print("\n=== Final Evaluation ===")
results = trainer.evaluate()
print(results)

# Detailed classification report
val_preds = trainer.predict(val_dataset)
pred_labels = np.argmax(val_preds.predictions, axis=-1)
print("\nClassification Report:")
print(classification_report(y_val, pred_labels,
                            target_names=["Safe", "Crisis"]))

# ─── CELL 10: Save model ──────────────────────────────────────────────────────
print(f"\n=== Saving model to {OUTPUT_DIR} ===")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save model metadata
import json
metadata = {
    "base_model": MODEL_NAME,
    "task": "binary_classification",
    "labels": {0: "safe", 1: "crisis"},
    "languages": ["am", "om", "ti", "en", "mixed"],
    "max_length": MAX_LENGTH,
    "training_samples": len(X_train),
    "validation_samples": len(X_val),
    "final_metrics": results,
}
with open(os.path.join(OUTPUT_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2, default=str)

print("Model saved! Download the 'ethiopian_crisis_model/' folder.")
print("Place it in Ethiopia/Model/ethiopian_crisis_model/ in your project.")
