from datasets import load_dataset, concatenate_datasets, ClassLabel, Dataset
from transformers import set_seed

import utils

"""
Divide the model training set like a MHAT algorithm out of the same public
"""


class DatasetPartition:

    def __init__(self, args):
        self.args = args
        self.private_datasets = []
        self.validation_datasets = []
        self.test_datasets = []

        self.public_datasets = []
        self.merged_val_datasets = []
        self.merged_test_datasets = []
        # No partitioning of the public dataset from the training dataset for the centralized or centralized_mixed algorithm
        self.centralized_train_datasets = []
        # Mixed training and validation dataset for the centralized_mixed algorithm
        self.mix_train_datasets = Dataset.from_dict({})
        self.mix_validation_datasets = Dataset.from_dict({})
        self.mix_test_datasets = Dataset.from_dict({})

        # ALL label statistics are in three categories (0:neural 1:positive 2:negative)
        # All datasets modified to two columns（label and text）for easy splicing public_dataset
        if self.args.algorithm not in ['centralized_mixed']:
            for dataset in args.datasets:
                raw_datasets = load_dataset(args.data_dir + '/{}'.format(dataset))
                raw_datasets = raw_datasets.remove_columns(['idx'])

                if dataset == 'automotive':
                    # convert label value in datasets
                    # raw_datasets = raw_datasets.map(utils.replace_labels)
                    # label type: Value ---> ClassLabel
                    raw_datasets = raw_datasets.class_encode_column('label')
                    automotive_train_dataset = raw_datasets['train']
                    automotive_val_dataset = raw_datasets['validation']
                    automotive_test_dataset = raw_datasets['test']

                    # Split public dataset
                    automotive_public_private_dataset = automotive_train_dataset.train_test_split(test_size=args.public_ratio, seed=args.seed, stratify_by_column='label')
                    automotive_private_dataset = automotive_public_private_dataset['train']
                    automotive_public_dataset = automotive_public_private_dataset['test']
                    del automotive_public_private_dataset

                    self.public_datasets.append(automotive_public_dataset)
                    self.private_datasets.append(automotive_private_dataset)
                    self.validation_datasets.append(automotive_val_dataset)
                    self.test_datasets.append(automotive_test_dataset)

                elif dataset == 'baby':
                    # convert label value in datasets
                    # raw_datasets = raw_datasets.map(utils.replace_labels)
                    # label type: Value ---> ClassLabel
                    raw_datasets = raw_datasets.class_encode_column('label')
                    baby_train_dataset = raw_datasets['train']
                    baby_val_dataset = raw_datasets['validation']
                    baby_test_dataset = raw_datasets['test']

                    # Split public dataset
                    baby_public_private_dataset = baby_train_dataset.train_test_split(test_size=args.public_ratio, seed=args.seed, stratify_by_column='label')
                    baby_private_dataset = baby_public_private_dataset['train']
                    baby_public_dataset = baby_public_private_dataset['test']
                    del baby_public_private_dataset

                    self.public_datasets.append(baby_public_dataset)
                    self.private_datasets.append(baby_private_dataset)
                    self.validation_datasets.append(baby_val_dataset)
                    self.test_datasets.append(baby_test_dataset)

                elif dataset == 'clothing':
                    # convert label value in datasets
                    # raw_datasets = raw_datasets.map(utils.replace_labels)
                    # label type: Value ---> ClassLabel
                    raw_datasets = raw_datasets.class_encode_column('label')
                    clothing_train_dataset = raw_datasets['train']
                    clothing_val_dataset = raw_datasets['validation']
                    clothing_test_dataset = raw_datasets['test']

                    # Split public dataset
                    clothing_public_private_dataset = clothing_train_dataset.train_test_split(test_size=args.public_ratio, seed=args.seed, stratify_by_column='label')
                    clothing_private_dataset = clothing_public_private_dataset['train']
                    clothing_public_dataset = clothing_public_private_dataset['test']
                    del clothing_public_private_dataset

                    self.public_datasets.append(clothing_public_dataset)
                    self.private_datasets.append(clothing_private_dataset)
                    self.validation_datasets.append(clothing_val_dataset)
                    self.test_datasets.append(clothing_test_dataset)

                elif dataset == 'health':
                    # convert label value in datasets
                    # raw_datasets = raw_datasets.map(utils.replace_labels)
                    # label type: Value ---> ClassLabel
                    raw_datasets = raw_datasets.class_encode_column('label')
                    health_train_dataset = raw_datasets['train']
                    health_val_dataset = raw_datasets['validation']
                    health_test_dataset = raw_datasets['test']

                    # Split public dataset
                    health_public_private_dataset = health_train_dataset.train_test_split(test_size=args.public_ratio,
                                                                                            seed=args.seed,
                                                                                            stratify_by_column='label')
                    health_private_dataset = health_public_private_dataset['train']
                    health_public_dataset = health_public_private_dataset['test']
                    del health_public_private_dataset

                    self.public_datasets.append(health_public_dataset)
                    self.private_datasets.append(health_private_dataset)
                    self.validation_datasets.append(health_val_dataset)
                    self.test_datasets.append(health_test_dataset)

                elif dataset == 'sport':
                    # convert label value in datasets
                    # raw_datasets = raw_datasets.map(utils.replace_labels)
                    # label type: Value ---> ClassLabel
                    raw_datasets = raw_datasets.class_encode_column('label')
                    sport_train_dataset = raw_datasets['train']
                    sport_val_dataset = raw_datasets['validation']
                    sport_test_dataset = raw_datasets['test']

                    # Split public dataset
                    sport_public_private_dataset = sport_train_dataset.train_test_split(test_size=args.public_ratio,
                                                                                          seed=args.seed,
                                                                                          stratify_by_column='label')
                    sport_private_dataset = sport_public_private_dataset['train']
                    sport_public_dataset = sport_public_private_dataset['test']
                    del sport_public_private_dataset

                    self.public_datasets.append(sport_public_dataset)
                    self.private_datasets.append(sport_private_dataset)
                    self.validation_datasets.append(sport_val_dataset)
                    self.test_datasets.append(sport_test_dataset)

            # Merge all public、validation and test data
            self.public_datasets = concatenate_datasets(self.public_datasets).shuffle(seed=self.args.seed)
            self.merged_val_datasets = concatenate_datasets(self.validation_datasets).shuffle(seed=self.args.seed)
            self.merged_test_datasets = concatenate_datasets(self.test_datasets).shuffle(seed=self.args.seed)

        # centralized_mixed dataset
        else:
            for dataset in args.datasets:
                raw_datasets = load_dataset(args.data_dir + '/{}'.format(dataset))
                raw_datasets = raw_datasets.remove_columns(['idx'])

                if dataset == 'automotive':
                    # raw_datasets = raw_datasets.map(utils.replace_labels)
                    # label type: Value ---> ClassLabel
                    raw_datasets = raw_datasets.class_encode_column('label')
                    automotive_train_dataset = raw_datasets['train']
                    automotive_val_dataset = raw_datasets['validation']
                    automotive_test_dataset = raw_datasets['test']

                    # Split public dataset
                    automotive_public_private_dataset = automotive_train_dataset.train_test_split(test_size=args.public_ratio,
                                                                                            seed=args.seed,
                                                                                            stratify_by_column='label')
                    automotive_private_dataset = automotive_public_private_dataset['train']
                    del automotive_public_private_dataset

                    self.centralized_train_datasets.append(automotive_private_dataset)
                    self.validation_datasets.append(automotive_val_dataset)
                    self.test_datasets.append(automotive_test_dataset)

                if dataset == 'baby':
                    # raw_datasets = raw_datasets.map(utils.replace_labels)
                    # label type: Value ---> ClassLabel
                    raw_datasets = raw_datasets.class_encode_column('label')
                    baby_train_dataset = raw_datasets['train']
                    baby_val_dataset = raw_datasets['validation']
                    baby_test_dataset = raw_datasets['test']

                    # Split public dataset
                    baby_public_private_dataset = baby_train_dataset.train_test_split(
                        test_size=args.public_ratio,
                        seed=args.seed,
                        stratify_by_column='label')
                    baby_private_dataset = baby_public_private_dataset['train']
                    del baby_public_private_dataset

                    self.centralized_train_datasets.append(baby_private_dataset)
                    self.validation_datasets.append(baby_val_dataset)
                    self.test_datasets.append(baby_test_dataset)

                if dataset == 'clothing':
                    # raw_datasets = raw_datasets.map(utils.replace_labels)
                    # label type: Value ---> ClassLabel
                    raw_datasets = raw_datasets.class_encode_column('label')
                    clothing_train_dataset = raw_datasets['train']
                    clothing_val_dataset = raw_datasets['validation']
                    clothing_test_dataset = raw_datasets['test']

                    # Split public dataset
                    clothing_public_private_dataset = clothing_train_dataset.train_test_split(test_size=args.public_ratio,
                                                                                            seed=args.seed,
                                                                                            stratify_by_column='label')
                    tsa_private_dataset = clothing_public_private_dataset['train']
                    del clothing_public_private_dataset

                    self.centralized_train_datasets.append(tsa_private_dataset)
                    self.validation_datasets.append(clothing_val_dataset)
                    self.test_datasets.append(clothing_test_dataset)

                if dataset == 'health':
                    # raw_datasets = raw_datasets.map(utils.replace_labels)
                    # label type: Value ---> ClassLabel
                    raw_datasets = raw_datasets.class_encode_column('label')
                    health_train_dataset = raw_datasets['train']
                    health_val_dataset = raw_datasets['validation']
                    health_test_dataset = raw_datasets['test']

                    # Split public dataset
                    health_public_private_dataset = health_train_dataset.train_test_split(
                        test_size=args.public_ratio,
                        seed=args.seed,
                        stratify_by_column='label')
                    health_private_dataset = health_public_private_dataset['train']
                    del health_public_private_dataset

                    self.centralized_train_datasets.append(health_private_dataset)
                    self.validation_datasets.append(health_val_dataset)
                    self.test_datasets.append(health_test_dataset)

                if dataset == 'sport':
                    # raw_datasets = raw_datasets.map(utils.replace_labels)
                    # label type: Value ---> ClassLabel
                    raw_datasets = raw_datasets.class_encode_column('label')
                    sport_train_dataset = raw_datasets['train']
                    sport_val_dataset = raw_datasets['validation']
                    sport_test_dataset = raw_datasets['test']

                    # Split public dataset
                    sport_public_private_dataset = sport_train_dataset.train_test_split(
                        test_size=args.public_ratio,
                        seed=args.seed,
                        stratify_by_column='label')
                    sport_private_dataset = sport_public_private_dataset['train']
                    del sport_public_private_dataset

                    self.centralized_train_datasets.append(sport_private_dataset)
                    self.validation_datasets.append(sport_val_dataset)
                    self.test_datasets.append(sport_test_dataset)

            self.mix_train_datasets = concatenate_datasets(self.centralized_train_datasets).shuffle(seed=self.args.seed)
            self.mix_validation_datasets = concatenate_datasets(self.validation_datasets).shuffle(seed=self.args.seed)
            self.mix_test_datasets = concatenate_datasets(self.test_datasets).shuffle(seed=self.args.seed)


if __name__ == '__main__':
    class args:
        seed = 42
        datasets = ['automotive', 'baby', 'clothing', 'health', 'sport']
        data_dir = '/gemini/data-1/non-iid-extracted-data'
        K = 5
        public_ratio = 0.2
        algorithm = 'centralized_mixed'


    set_seed(args.seed)

    sa = DatasetPartition(args)
    train_datasets = sa.private_datasets
    public_dataset = sa.public_datasets
    validation_datasets = sa.validation_datasets
    test_datasets = sa.test_datasets

    merged_val_dataset = sa.merged_val_datasets
    merged_test_dataset = sa.merged_test_datasets

    centralized_train_datasets = sa.centralized_train_datasets
    mix_train_datasets = sa.mix_train_datasets
    mix_validation_datasets = sa.mix_validation_datasets
    mix_test_datasets = sa.mix_test_datasets

    if centralized_train_datasets:
        print('length of centralized_train_datasets: {}'.format(
            [len(centralized_train_datasets[k]) for k in range(args.K)]))
    else:
        if public_dataset:
            print('length of public_dataset: {}'.format(len(public_dataset)))
        if train_datasets:
            print('length of train_datasets: {}'.format([len(train_datasets[k]) for k in range(args.K)]))
        if merged_val_dataset:
            print('length of merged_val_dataset: {}'.format(len(merged_val_dataset)))
        if merged_test_dataset:
            print('length of merged_test_dataset: {}'.format(len(merged_test_dataset)))
    if validation_datasets:
        print('length of val_datasets: {}'.format([len(validation_datasets[k]) for k in range(args.K)]))
    if test_datasets:
        print('length of test_datasets: {}'.format([len(test_datasets[k]) for k in range(args.K)]))
    if mix_train_datasets:
        print('length of mix_train_datasets: {}'.format(len(mix_train_datasets)))
    if mix_validation_datasets:
        print('length of mix_validation_datasets: {}'.format(len(mix_validation_datasets)))
    if mix_test_datasets:
        print('length of mix_test_datasets: {}'.format(len(mix_test_datasets)))