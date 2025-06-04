import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import numpy as np
import evaluate

# Load the parquet file
df = pd.read_parquet("ipj/irish-penny.parquet")

# Create dataset format: each row becomes 2 entries
data = []
for idx, row in df.iterrows():
    # Keep pairs together by adding them consecutively
    data.append({"text": row["cleaned_text"], "label": 1})
    data.append({"text": row["modernized_text"], "label": 0})

# Convert to HuggingFace Dataset
dataset = Dataset.from_list(data)

# Split into train/test (keeping pairs together by using even indices)
train_size = int(0.95 * len(dataset) / 2) * 2  # Ensure even number
train_dataset = dataset.select(range(train_size))
test_dataset = dataset.select(range(train_size, len(dataset)))

# Load tokenizer and model
model_name = "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Tokenize function
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


# Tokenize datasets
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "modernized", 1: "cleaned"},
    label2id={"modernized": 0, "cleaned": 1},
)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_score = f1.compute(predictions=predictions, references=labels)

    return {"accuracy": acc["accuracy"], "f1": f1_score["f1"]}


# Training arguments with STEP-based evaluation
training_args = TrainingArguments(
    output_dir="./checkpoints",
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    # Learning rate schedule configuration
    lr_scheduler_type="cosine",
    warmup_steps=200,
    # warmup_ratio=0.1,
    push_to_hub=False,
)


# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# Train
trainer.train()

# Save model
trainer.save_model("./ipj_classifier")

## Test inference
# from transformers import pipeline
# classifier = pipeline("sentiment-analysis", model="./final_model")
#
## Example usage
# test_text = "Your example text here"
# result = classifier(test_text)
# print(f"Text: {test_text}")
# print(f"Prediction: {result}")
