import logging
import os

import pandas as pd
import torch
import evaluate
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import preprocessing_raw_datasets


def validation(node, validation_dataset):
    node.model.to(node.device).eval()

    validation_dataset = preprocessing_raw_datasets(validation_dataset, node.tokenizer, node.max_seq_length)
    validation_dataloader = DataLoader(validation_dataset, shuffle=False, batch_size=node.batch_size, collate_fn=node.data_collator)

    eval_loss = 0.
    metric = evaluate.load(node.args.metric_type)
    for batch in tqdm(validation_dataloader, desc='Evaluating'):
        batch = {k: v.to(node.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = node.model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        eval_loss += loss
        if node.args.num_classes == 1:
            prediction = logits.squeeze()
        else:
            prediction = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=prediction, references=batch['labels'])

    eval_loss = eval_loss / len(validation_dataloader)
    eval_results = metric.compute(average='weighted')

    node.model.cpu()
    return eval_loss, eval_results


def test(node, test_dataset):
    node.model.to(node.device).eval()

    test_dataset = preprocessing_raw_datasets(test_dataset, node.tokenizer, node.max_seq_length)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=node.batch_size, collate_fn=node.data_collator)

    test_loss = 0.
    metric = evaluate.load(node.args.metric_type)
    for batch in tqdm(test_dataloader, desc='Predicting'):
        batch = {k: v.to(node.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = node.model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        test_loss += loss
        if node.args.num_classes == 1:
            prediction = logits.squeeze()
        else:
            prediction = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=prediction, references=batch['labels'])

    test_loss = test_loss / len(test_dataloader)
    test_results = metric.compute(average='weighted')

    node.model.cpu()
    return test_loss, test_results
    

class Recorder:
    def __init__(self, args):
        self.args = args

        self.test_acc = pd.DataFrame(columns=range(args.K + 1))
        self.val_acc = pd.DataFrame(columns=range(args.K + 1))
        self.current_test_acc = {k: None for k in range(args.K + 6)}
        self.current_acc = {k: None for k in range(args.K + 6)}
        self.best_acc = torch.zeros(self.args.K + 1)
        self.get_a_better = torch.zeros(self.args.K + 1)

    def evaluate(self, node, validation_dataset):
        val_loss, val_results = validation(node, validation_dataset)
        logging.info('val_loss={}, val_results={}'.format(val_loss, val_results))

        self.current_acc[node.id] = '{:.1f}'.format(val_results['f1'] * 100)

    def predict(self, node, test_dataset):
        # output_eval_dir = os.path.join(self.args.submission_dir, '{}_{}'.format(node.name, node.model_type))
        # os.makedirs(output_eval_dir, exist_ok=True)
        test_loss, test_results = test(node, test_dataset)
        logging.info('test_loss={}, test_results={}'.format(test_loss, test_results))

        self.current_test_acc[node.id] = '{:.1f}'.format(test_results['f1'] * 100)

    def save_model(self, node):
        model_to_save = node.model.module if hasattr(node.model, 'module') else node.model
        model_type = node.model_type if "/" not in node.model_type else node.model_type.split("/")[-1]
        file_name = os.path.join(self.args.model_dir, self.args.algorithm, '{}_{}'.format(node.name, model_type))
        model_to_save.save_pretrained(file_name)
        node.tokenizer.save_pretrained(file_name)

    def save_record(self, node=None, row=None, col=None, round_=None):
        # centralized_mixed algorithm record the values
        if self.args.algorithm == 'centralized_mixed':
            self.val_acc.loc[len(self.val_acc)] = self.current_acc
            print(f'validation values: \n {self.val_acc}')
            self.val_acc.to_csv(os.path.join(self.args.record_dir, '{}_dev.csv'.format(self.args.algorithm)))
            # Record the test values
            if self.args.do_test:
                self.test_acc.loc[len(self.test_acc)] = self.current_test_acc
                print(f'test values: \n {self.test_acc}')
                self.test_acc.to_csv(os.path.join(self.args.submission_dir, '{}_test.csv'.format(self.args.algorithm)))
        else:
            # Calculate the value of row of mhat algorithm
            if round_ is not None:
                row_dev = row
                row_dev += round_ * self.args.dis_epochs
                self.val_acc.at[row_dev, col] = self.current_acc[node.id]
            else:
                self.val_acc.at[row, col] = self.current_acc[node.id]
            print(f'validation values: \n {self.val_acc}')
            self.val_acc.to_csv(os.path.join(self.args.record_dir, '{}_dev.csv'.format(self.args.algorithm)))
            # Record the test values
            if self.args.do_test:
                if round_ is not None:
                    row_test = row
                    row_test += round_ * self.args.dis_epochs
                    self.test_acc.at[row_test, col] = self.current_test_acc[node.id]
                else:
                    self.test_acc.at[row, col] = self.current_test_acc[node.id]
                print(f'test values: \n {self.test_acc}')
                self.test_acc.to_csv(
                    os.path.join(self.args.submission_dir, '{}_test.csv'.format(self.args.algorithm)))
        