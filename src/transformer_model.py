from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import torch
from torch.utils.data import Dataset
import json

class SuicideDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_length)
        self.labels = [0 if l=="non suicide" else (1 if l=="depression" else 2) for l in labels]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def run_transformer(X_train, X_test, y_train, y_test, model_name="bert-base-uncased", out_path="results/transformer_metrics.json"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = SuicideDataset(X_train, y_train, tokenizer)
    test_dataset = SuicideDataset(X_test, y_test, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    per_device_train_batch_size=16,
    num_train_epochs=2,
    save_total_limit=1,
    logging_dir='./logs',
    logging_strategy="steps",
    logging_steps=50)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()
    metrics = trainer.evaluate()
    with open(out_path, "w") as f:
        json.dump(metrics, f)
    print(metrics)