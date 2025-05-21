import logging
import sys
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler


def compute_entropy(logits):
    # 1. Softmax
    probabilities = torch.softmax(logits, 1)

    # 2. Calculate the entropy of each piece of data
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)

    # 3. Average
    average_entropy = torch.mean(entropy)

    return average_entropy.item()
    

def softmax_with_temperature(logits, temperature):
    if temperature <= 0:
        temperature = 1
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=-1)


def calculate_linear_inverse_proportions(weights):
    # Avoid weights being zero
    epsilon = 1e-6
    # Calculate the inverse of each weight
    inverse_weights = [1.0 / (weight + epsilon) for weight in weights]
    total_inverse = sum(inverse_weights)
    proportions = [inverse_weight / total_inverse for inverse_weight in inverse_weights]

    return proportions


def calculate_softmax_inverse_proportions(weights, alpha, client_num, epochs):
    # Avoid weights being zero
    epsilon = 1e-6
    # Calculate the inverse exponential function for each weight
    if alpha > 0:
        inverse_weights = [np.exp(-(weight + epsilon) * alpha) for weight in weights]
    else:
        exit('Factor alpha should be positive number!')

    # Weights in the same round are normalized
    if len(weights) == len(client_num):
        total_inverse = sum(inverse_weights)
        proportions = [inverse_weight / total_inverse for inverse_weight in inverse_weights]
    else:
        total_inverse_weights = [0.0 for _ in range(epochs)]
        for k in range(len(inverse_weights)):
            total_inverse_weights[k % epochs] += inverse_weights[k]
        proportions = [inverse_weights[i] / total_inverse_weights[i % epochs] for i in range(len(inverse_weights))]
        logging.info('total_sum_weights: {}'.format(total_inverse_weights))

    return proportions


# Convert label value in datasets （0-1：neg(2); 2：neural(0); 3-4：pos(1)）
def replace_labels(dataset):
    if dataset['label'] == 2:
        dataset['label'] = 0
    elif dataset['label'] in [3, 4]:
        dataset['label'] = 1
    elif dataset['label'] in [0, 1]:
        dataset['label'] = 2
    return dataset


class IgnoreSpecificMessageFilter(logging.Filter):
    def filter(self, record):
        if 'Loading cached processed dataset' in record.getMessage():
            return False
        return True


def swap_text_label_column(dataset):
    dataset = dataset.rename_column('text', 'temp_text')
    dataset = dataset.rename_column('label', 'temp_label')

    # Swap text and label columns
    dataset = dataset.map(lambda example: {'label': example['temp_label'], 'text': example['temp_text']})
    dataset = dataset.remove_columns(['temp_text', 'temp_label'])

    return dataset


def init_model(name, model_type, num_classes):
    # config = AutoConfig.from_pretrained(model_type, num_labels=num_classes, finetuning_task=task_name)
    tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    total_params = total_params / 1000000
    logging.info('model parameters of %s_%s: %2.1fM' % (name, model_type, total_params))
    return tokenizer, model


def init_optimizer(optimizer_type, model, lr, weight_decay=0., momentum=0.9):
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(momentum, 0.999),
                                     eps=1e-8)
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(momentum, 0.999),
                                      eps=1e-8)
    else:
        sys.exit("Not implemented optimizer, code exit, re-run to use correct optimizer")
    return optimizer


def init_scheduler(scheduler_type, optimizer, num_warmup_steps=None, num_training_steps=None):
    if scheduler_type == 'linear':
        scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=num_warmup_steps,
                                  num_training_steps=num_training_steps)
    elif scheduler_type == "cosine":  # cosine
        scheduler = get_scheduler('cosine', optimizer=optimizer, num_warmup_steps=num_warmup_steps,
                                  num_training_steps=num_training_steps)
    else:
        sys.exit("Not implemented learning rate scheduler, code exit, re-run to use correct scheduler")
    return scheduler


def preprocessing_raw_datasets(raw_dataset, tokenizer, max_seq_length, logits=None):
    text_key = 'text'

    def preprocess_function(text):
        return tokenizer(text[text_key], padding=True, max_length=max_seq_length, truncation=True)

    # Define a filter fun to remove blank lines
    def filter_empty_rows(example):
        return example[text_key] is not None and example[text_key].strip() != ''

    # Apply a filter function to each data segmentation
    raw_dataset = raw_dataset.filter(filter_empty_rows)
    encoded_dataset = raw_dataset.map(preprocess_function, batched=True)
    # The cause of the error is that the tokenized datasets object has columns with strings, and the data collator does not know how to pad these
    encoded_dataset = encoded_dataset.remove_columns(text_key)
    encoded_dataset = encoded_dataset.rename_column('label', 'labels')

    if logits is not None:
        if type(logits) == list:
            for k in range(len(logits)):
                encoded_dataset = encoded_dataset.add_column('logits{}'.format(k), logits[k].tolist())
        else:
            encoded_dataset = encoded_dataset.add_column('logits', logits.tolist())

    return encoded_dataset


if __name__ == '__main__':
    raw_datasets = load_dataset('./data/yr')['train']
    print(raw_datasets)

    # loading BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # def tokenize_function(example):
    #     return tokenizer(example['text'],padding=True, max_length=128, truncation=True)
    #
    #
    # tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    # tokenized_datasets = tokenized_datasets.remove_columns('text')
    # tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # tokenized_datasets.set_format("torch")
    tokenized_datasets = preprocessing_raw_datasets(raw_datasets, tokenizer, 128)
    print(tokenized_datasets)
    for i in range(5):
        print(f"Example {i}:")
        print(f"Input IDs: {tokenized_datasets[i]['input_ids']}")
        print(f"Attention Mask: {tokenized_datasets[i]['attention_mask']}")
