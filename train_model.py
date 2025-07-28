import os
import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

# Load your financial news dataset
df = pd.read_csv('financial_news.csv')

# Convert string labels to numerical labels
label_mapping = {'positive': 0, 'neutral': 1, 'negative': 2}
df['label'] = df['label'].map(label_mapping)

# Convert the DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Split the dataset into train and test sets
dataset = dataset.train_test_split(test_size=0.2)

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load the model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('financial_sentiment_model')
tokenizer.save_pretrained('financial_sentiment_model')