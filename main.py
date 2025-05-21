import argparse
import logging
import sys

import torch
import os
import numpy as np
from transformers import set_seed

import utils
from dataset import DatasetPartition
from recorder import Recorder
from node import Client, Server
from gpt import get_weights_by_gpt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Common Hyper-parameters
    # Total
    parser.add_argument('--algorithm', type=str, default='fed_tld',
                        help='Type of algorithms: {centralized, centralized_mixed, fed_avg, fed_kd, fed_max, mhat_ce, mhat_kl, fed_tld}')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--K', type=int, default=5, help='Number of clients')
    parser.add_argument('--C', type=float, default=1, help='Fraction of clients')
    parser.add_argument('--R', type=int, default=5, help='Number of rounds')

    # Data
    parser.add_argument('--metric_type', type=str, default='/gemini/code/evaluate/metrics/f1',
                        help='Metric used to evaluate the model')
    parser.add_argument('--datasets', type=str, default=['automotive', 'baby', 'clothing', 'health', 'sport'],
                        help='Type of dataset: [automotive, baby, clothing, health, sport]')
    parser.add_argument('--data_dir', type=str, default='/gemini/data-1/non-iid-extracted-data', help='Path of data dir')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='the all dataset(automotive, baby, clothing, health, sport) is divided into three categories')
    parser.add_argument('--public_ratio', type=float, default=0.2, help='Ratio of public dataset')
    # Server Model
    parser.add_argument('--central_model', type=str, default='/gemini/code/model/roberta-large',
                        help='Type of global model: {bert-base-uncased, bert-large-uncased, roberta-base, roberta-large, xlnet-large-cased}')

    # Output
    parser.add_argument("--output_dir", type=str, default="/gemini/output",
                        help="The output directory where checkpoints/results/logs will be written.")
    parser.add_argument('--do_test', action='store_true', default=True,
                        help='Whether to make predictions at the end of the last round.')

    # Specific Hyper-parameters
    # Data
    parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    # Local Model
    parser.add_argument('--local_models', type=str,
                        default='/gemini/code/model/bert-base-cased,/gemini/code/model/bert-large-cased,/gemini/code/model/roberta-base,/gemini/code/model/roberta-large,/gemini/code/model/xlnet-large-cased',
                        help='Type of local model: {bert-base-cased, bert-large-cased, roberta-base, roberta-large, xlnet-large-cased}')
    # Optima
    parser.add_argument('--E', type=int, default=3, help='Number of local epochs')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Type of optimizer: {sgd, adam, adamw}')
    parser.add_argument('--scheduler', type=str, default='linear', help='Type of scheduler: {liner, cosine}')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate of local training')
    parser.add_argument('--weight_decay', type=float, default=0, help="Weight decay for optimizer")
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum for optimizer')
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Step of training to perform learning rate warmup for if set for cosine and linear decay")
    parser.add_argument('--temperature', type=float, default=1,
                        help='Temperature must be a positive value, it is best in the range (0, 10]')

    # Distillation args
    parser.add_argument('--dis_epochs', type=int, default=3, help='Number of distillation epochs')
    parser.add_argument('--dis_lr', type=float, default=2e-5, help='Learning rate of distillation ')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Set device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set output_dir
    args.model_dir = os.path.join(args.output_dir, args.algorithm, 'model')
    args.record_dir = os.path.join(args.output_dir, args.algorithm, 'record')
    args.log_dir = os.path.join(args.output_dir, args.algorithm, 'log')
    args.submission_dir = os.path.join(args.output_dir, args.algorithm, 'submission')
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.record_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.submission_dir, exist_ok=True)

    # Set log
    logger = logging.getLogger(__name__)
    filter = utils.IgnoreSpecificMessageFilter()
    fh = logging.FileHandler(os.path.join(args.log_dir, '{}.txt'.format(args.algorithm)))
    sh = logging.StreamHandler(sys.stdout)
    fh.addFilter(filter)
    sh.addFilter(filter)
    logging.basicConfig(format="[%(levelname)s](%(asctime)s) %(message)s",
                        level=logging.INFO,
                        datefmt="%m/%d/%Y %H:%M:%S",
                        handlers=[fh, sh])

    # Set data
    sa = DatasetPartition(args)
    # local model's private data
    private_datasets = sa.private_datasets
    # public data
    public_datasets = sa.public_datasets
    # validation and test data
    validation_datasets = sa.validation_datasets
    test_datasets = sa.test_datasets

    # merged all validation and test data for validating and testing mhat
    merged_val_datasets = sa.merged_val_datasets
    merged_test_datasets = sa.merged_test_datasets

    # centralized training dataset, no partitioning of the public dataset from the training dataset
    centralized_train_datasets = sa.centralized_train_datasets
    # mixed training and validation dataset for centralized_mixed algorithm
    mix_train_datasets = sa.mix_train_datasets
    mix_validation_datasets = sa.mix_validation_datasets
    mix_test_datasets = sa.mix_test_datasets

    # Set recoder
    recorder = Recorder(args)

    # Federated training
    logger.info('Running on %s', args.device)
    logger.info("algorithm: {}".format(args.algorithm))
    logger.info("dataset: {},\tpublic_ratio: {}".format(args.datasets, args.public_ratio, ))
    if centralized_train_datasets:
        logger.info('length of centralized_train_datasets: {}'.format(
            [len(centralized_train_datasets[k]) for k in range(args.K)]))
    else:
        if public_datasets:
            logger.info('length of public_dataset: {}'.format(len(public_datasets)))
        if private_datasets:
            logger.info('length of train_datasets: {}'.format([len(private_datasets[k]) for k in range(args.K)]))
        if merged_val_datasets:
            logger.info('length of merged_val_dataset: {}'.format(len(merged_val_datasets)))
        if merged_test_datasets:
            logger.info('length of merged_test_dataset: {}'.format(len(merged_test_datasets)))
    if validation_datasets:
        logger.info('length of val_datasets: {}'.format([len(validation_datasets[k]) for k in range(args.K)]))
    if test_datasets:
        logger.info('length of test_datasets: {}'.format([len(test_datasets[k]) for k in range(args.K)]))
    if mix_train_datasets:
        logger.info('length of mix_train_datasets: {}'.format(len(mix_train_datasets)))
    if mix_validation_datasets:
        logger.info('length of mix_validation_datasets: {}'.format(len(mix_validation_datasets)))
    if mix_test_datasets:
        logger.info('length of mix_test_datasets: {}'.format(len(mix_test_datasets)))

    logger.info("num_clients: {},\tfraction: {}".format(args.K, args.C))
    logger.info("global_rounds: {}".format(args.R))
    logger.info("global_model: {}".format(args.central_model))
    args.local_models = args.local_models.split(',')
    logger.info("local_models: [{}]".format(', '.join(args.local_models)))
    logger.info(
        "batch_size: {},\tmax_seq_length: {},\tlocal_epochs: {},"
        "\tlr: {},\ttemperature: {}".format(args.batch_size, args.max_seq_length, args.E, args.lr, args.temperature))
    if args.algorithm not in ['centralized', 'centralized_mixed']:
        logger.info("distillation_epochs: {},\tdistillation_lr: {}".format(args.dis_epochs, args.dis_lr))
    
    # Initial server
    server = Server(args, id=0, model_type=args.central_model, public_dataset=public_datasets)
    # Initial clients
    if args.algorithm not in ['centralized_mixed']:
        clients = {
            k + 1: Client(args, id=k + 1, model_type=args.local_models[k], train_dataset=private_datasets[k])
            for k in range(args.K)}
    else:
        clients = {k + 1: Client(args, id=k + 1, model_type=args.local_models[k]) for k in range(args.K)}

    if args.algorithm == 'centralized_mixed':
        logger.info("# Node{:d}: {}_{}".format(server.id, server.name, server.model_type))

        # local training numbers = FL's number of rounds * local epochs
        for epoch in range(args.E * args.R):
            logging.info('Epoch {}/{}: '.format(epoch + 1, args.E * args.R))
            # Mixed full train_dataset training
            server.centralized_mixed_training(mix_train_datasets)
            # Mixed full val_dataset validation
            recorder.evaluate(server, mix_validation_datasets)
            # Mixed full test_dataset testing
            recorder.predict(server, mix_test_datasets)

            # separate testing on different test_datasets
            for k in range(len(args.datasets)):
                server.id += 1
                # Predict server!
                if args.do_test:
                    recorder.predict(server, test_datasets[k])
            server.id = 0

            # Save record!
            recorder.save_record()
        # Save server!
        recorder.save_model(server)

    elif args.algorithm == 'fed_avg':
        # randomly sample partial clients: m = max(C*K, 1)
        m = max(int(args.C * args.K), 1)
        cur_selected_clients = sorted(np.random.choice(range(1, args.K + 1), m, replace=False))

        for round_ in range(args.R):
            logger.info('===============The {:d}-th round==============='.format(round_ + 1))

            weights = []
            logits_locals = []
            for k in cur_selected_clients:
                # ClientExecute()
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))
                for epoch in range(args.E):
                    logging.info('Epoch {}/{}: '.format(epoch + 1, args.E))

                    """ 1. Local Training """
                    # θ_t^k ← ClientUpdate(D^k; θ_t-1^k)
                    client.local_update()
                    # Eval each client!
                    recorder.evaluate(client, validation_datasets[k - 1])
                    # Predict each client and record the loss of test as client's weight
                    if args.do_test:
                        recorder.predict(client, test_datasets[k - 1])
                """ 2. Local Prediction """
                # Compute local logits: Y_t^k ← f^k(X^0; θ_t^k)
                logits = client.compute_logits(public_datasets)

                """ 3. Upload """
                # Upload local logits
                logits_locals.append(logits)
                weights = [1 / len(cur_selected_clients) for _ in cur_selected_clients]

            logger.info("Clients assigned by weight: {}".format(weights))

            """ 4. Aggregation """
            logits_glob = server.logit_ensemble(logits_locals, weights)

            # ServerExecute()
            logger.info("# Node{:d}: {}_{}".format(server.id, server.name, server.model_type))
            """ 5. Server Distillation """
            for epoch in range(args.dis_epochs):
                logging.info('Distilling Epoch {}/{}:'.format(epoch + 1, args.dis_epochs))
                # ①server distillation: θ_t ← FedTLDistillation(D^0 ∪ {Y_t^1, Y_t^2, Y_t^3, Y_t^4, Y_t^5}; θ_t-1)
                server.ensemble_distillation(public_datasets, logits_glob)  # aggregated by the server model
                # Eval server!
                recorder.evaluate(server, merged_val_datasets)
                # Predict server!
                if args.do_test:
                    recorder.predict(server, merged_test_datasets)
                    recorder.save_record(server, epoch, server.id, round_)
                    # The server predicts each test set separately and columns 6-10 are the values under each test set
                    for i in range(len(args.datasets) + 1, len(args.datasets) + 6):
                        server.id = i
                        recorder.predict(server, test_datasets[i - args.K - 1])
                        # Save record of server's distillation!
                        recorder.save_record(server, epoch, server.id, round_)
                server.id = 0
            # Save server!
            recorder.save_model(server)

            """ 6. Server Aggregation """
            # Compute server logits as aggregation: Y_t ← f(X^0; θ_t)
            logits_glob = server.compute_logits(public_datasets)

            """ 7. Local Distillation """
            # ②client distillation: θ_t^k ← ClientUpdate((X^0, Y_t); θ_t^k)
            logger.info("______ clients have received the logits_glob _____")
            for k in cur_selected_clients:
                # ClientExecute()
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))
                for epoch in range(args.dis_epochs):
                    logging.info('Distilling Epoch {}/{}:'.format(epoch + 1, args.dis_epochs))

                    client.local_distillation(public_datasets, logits_glob)
                    # Eval each client!
                    recorder.evaluate(client, validation_datasets[k - 1])
                    # Predict each client!
                    if args.do_test:
                        recorder.predict(client, test_datasets[k - 1])

                    # Save record of client's distillation!
                    recorder.save_record(client, epoch, k, round_)
                # Save each client!
                recorder.save_model(client)

    elif args.algorithm == 'fed_kd':
        # randomly sample partial clients: m = max(C*K, 1)
        m = max(int(args.C * args.K), 1)
        cur_selected_clients = sorted(np.random.choice(range(1, args.K + 1), m, replace=False))

        # Get the quantity of clients joined in the FL train for updating the clients weights
        cur_tot_client_lens = 0
        for k in cur_selected_clients:
            cur_tot_client_lens += len(clients[k].train_dataset)  # 498+498+498+498+498=2490

        weights = []
        logits_locals = []
        for k in cur_selected_clients:
            # ClientExecute()
            client = clients[k]
            logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))
            for epoch in range(args.E):
                logging.info('Epoch {}/{}: '.format(epoch + 1, args.E))

                """ 1. Local Training """
                # θ_t^k ← ClientUpdate(D^k; θ_t-1^k)
                client.local_update()
                # Eval each client!
                recorder.evaluate(client, validation_datasets[k - 1])
                # Predict each client and record the loss of test as client's weight
                if args.do_test:
                    recorder.predict(client, test_datasets[k - 1])

            """ 2. Local Prediction """
            # Compute local logits
            logits = client.compute_logits(public_datasets)

            """ 3. Upload """
            # Upload local logits
            logits_locals.append(logits)
            weights.append(len(client.train_dataset) / cur_tot_client_lens)

        logger.info("Clients assigned by weight: {}".format(weights))

        # ServerExecute()
        logger.info("# Node{:d}: {}_{}".format(server.id, server.name, server.model_type))

        """ 4. Aggregation """
        logits_glob = server.logit_ensemble(logits_locals, weights)

        for epoch in range(args.dis_epochs):
            logging.info('Distilling Epoch {}/{}:'.format(epoch + 1, args.dis_epochs))
            """ 5. Distillation """
            # θ_t ← EnsembleDistillation((X^0, Y_t); θ_t-1)
            server.ensemble_distillation(public_datasets, logits_glob)
            # Eval server!
            recorder.evaluate(server, merged_val_datasets)
            # Predict server!
            if args.do_test:
                recorder.predict(server, merged_test_datasets)
                # Save record!
                recorder.save_record(server, epoch, server.id)
                # The server separated testing on each test datasets
                for k in range(len(args.datasets)):
                    server.id = k + 1
                    recorder.predict(server, test_datasets[k])
                    # Save record!
                    recorder.save_record(server, epoch, server.id)
            server.id = 0
        # Save server!
        recorder.save_model(server)

    elif args.algorithm in ['mhat_kl', 'mhat_ce']:
        # randomly sample partial clients: m = max(C*K, 1)
        m = max(int(args.C * args.K), 1)
        cur_selected_clients = sorted(np.random.choice(range(1, args.K + 1), m, replace=False))

        # Get the quantity of clients joined in the FL train for updating the clients weights
        cur_tot_dataset_lens = 0
        for k in cur_selected_clients:
            cur_tot_dataset_lens += len(clients[k].train_dataset)

        for round_ in range(args.R):
            logger.info('===============The {:d}-th round==============='.format(round_ + 1))

            weights = []
            logits_locals = []
            for k in cur_selected_clients:
                # ClientExecute()
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))
                for epoch in range(args.E):
                    logging.info('Epoch {}/{}: '.format(epoch + 1, args.E))

                    """ 1. Local Training """
                    # θ_t^k ← ClientUpdate(D^k; θ_t-1^k)
                    client.local_update()
                    # Eval each client!
                    recorder.evaluate(client, validation_datasets[k - 1])
                    # Predict each client and record the loss of test as client's weight
                    if args.do_test:
                        recorder.predict(client, test_datasets[k - 1])

                """ 2. Local Prediction """
                # Compute local logits: Y_t^k ← f^k(X^0; θ_t^k)
                logits = client.compute_logits(public_datasets)

                """ 3. Upload """
                # Upload local logits
                logits_locals.append(logits)
                weights.append(len(client.train_dataset) / cur_tot_dataset_lens)

            logger.info("Clients assigned by weight: {}".format(weights))

            # ServerExecute()
            logger.info("# Node{:d}: {}_{}".format(server.id, server.name, server.model_type))

            """ 5. Server Distillation """
            for epoch in range(args.dis_epochs):
                logging.info('Distilling Epoch {}/{}:'.format(epoch + 1, args.dis_epochs))
                # ①server distillation: θ_t ← FedTLDistillation(D^0 ∪ {Y_t^1, Y_t^2, Y_t^3, Y_t^4, Y_t^5}; θ_t-1)
                server.mhat_distillation(public_datasets, logits_locals, weights)  # aggregated by the server model
                # Eval server!
                recorder.evaluate(server, merged_val_datasets)
                # Predict server!
                if args.do_test:
                    recorder.predict(server, merged_test_datasets)
                    recorder.save_record(server, epoch, server.id, round_)
                    # The server predicts each test set separately and columns 6-10 are the values under each test set
                    for i in range(len(args.datasets) + 1, len(args.datasets) + 6):
                        server.id = i
                        recorder.predict(server, test_datasets[i - args.K - 1])
                        # Save record of server's distillation!
                        recorder.save_record(server, epoch, server.id, round_)
                server.id = 0
            # Save server!
            recorder.save_model(server)

            """ 6. Server Aggregation """
            # Compute server logits as aggregation: Y_t ← f(X^0; θ_t)
            logits_glob = server.compute_logits(public_datasets)

            """ 7. Local Distillation """
            # ②client distillation: θ_t^k ← ClientUpdate((X^0, Y_t); θ_t^k)
            logger.info("______ clients have received the logits_glob _____")
            for k in cur_selected_clients:
                # ClientExecute()
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))
                for epoch in range(args.dis_epochs):
                    logging.info('Distilling Epoch {}/{}:'.format(epoch + 1, args.dis_epochs))

                    client.local_distillation(public_datasets, logits_glob)
                    # Eval each client!
                    recorder.evaluate(client, validation_datasets[k - 1])
                    # Predict each client!
                    if args.do_test:
                        recorder.predict(client, test_datasets[k - 1])

                    # Save record of client's distillation!
                    recorder.save_record(client, epoch, k, round_)
                # Save each client!
                recorder.save_model(client)

    elif args.algorithm == 'ds_fl':
        # randomly sample partial clients: m = max(C*K, 1)
        m = max(int(args.C * args.K), 1)
        cur_selected_clients = sorted(np.random.choice(range(1, args.K + 1), m, replace=False))

        for round_ in range(args.R):
            logger.info('===============The {:d}-th round==============='.format(round_ + 1))

            weights = []
            logits_locals = []
            for k in cur_selected_clients:
                # ClientExecute()
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

                for epoch in range(args.E):
                    logging.info('Epoch {}/{}: '.format(epoch + 1, args.E))
                    """ 1. Local Training """
                    # θ_t^k ← ClientUpdate(D^k; θ_t-1^k)
                    client.local_update()
                    # Eval each client!
                    recorder.evaluate(client, validation_datasets[k - 1])
                    # Predict each client!
                    if args.do_test:
                        recorder.predict(client, test_datasets[k - 1])

                """ 2. Local Prediction """
                # Compute local logits: Y_t^k ← f^k(X^0; θ_t^k)
                logits = client.compute_logits(public_datasets)

                """ 3. Upload """
                # Upload local logits
                logits_locals.append(logits)
                weights = [1 / len(cur_selected_clients) for _ in cur_selected_clients]

            # ServerExecute()
            logger.info("# Node{:d}: {}_{}".format(server.id, server.name, server.model_type))

            """ 4. Aggregation (ERA) """
            logits_glob = server.logit_ensemble_with_ERA(logits_locals, weights)  # softmax(∑ |D^k| / |D| * Y_t^k / t)

            """ 5. Server Distillation """
            for epoch in range(args.dis_epochs):
                logging.info('Distilling Epoch {}/{}:'.format(epoch + 1, args.dis_epochs))
                # ①server distillation: θ_t ← ServerDistillation((X^0, Y_t); θ_t-1)
                server.ensemble_distillation(public_datasets, logits_glob)
                # Eval server!
                recorder.evaluate(server, merged_val_datasets)
                # Predict server!
                if args.do_test:
                    recorder.predict(server, merged_test_datasets)
                    recorder.save_record(server, epoch, server.id, round_)
                    # The server predicts each test set separately and columns 6-10 are the values under each test set
                    for i in range(len(args.datasets) + 1, len(args.datasets) + 6):
                        server.id = i
                        recorder.predict(server, test_datasets[i - args.K - 1])
                        # Save record of server's distillation!
                        recorder.save_record(server, epoch, server.id, round_)
                server.id = 0

            # Save the server!
            recorder.save_model(server)

            """ 6. Local Distillation """
            # ②client distillation: θ_t^k ← ClientDistillation(X^0 ∪ Y_t; θ_t^k)
            for k in cur_selected_clients:
                # ClientExecute()
                client = clients[k]

                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))
                for epoch in range(args.dis_epochs):
                    logging.info('Distilling Epoch {}/{}:'.format(epoch + 1, args.dis_epochs))

                    client.local_distillation(public_datasets, logits_glob)
                    # Eval server on each dev set!
                    recorder.evaluate(client, validation_datasets[k - 1])
                    # Predict server on each test set!
                    if args.do_test:
                        recorder.predict(client, test_datasets[ k - 1])
                
                    # Save record of client's distillation!
                    recorder.save_record(client, epoch, k, round_)
                # Save each client!
                recorder.save_model(client)

    elif args.algorithm == 'fed_max':
        # randomly sample partial clients: m = max(C*K, 1)
        m = max(int(args.C * args.K), 1)
        cur_selected_clients = sorted(np.random.choice(range(1, args.K + 1), m, replace=False))

        for round_ in range(args.R):
            logger.info('===============The {:d}-th round==============='.format(round_ + 1))

            # Minimum loss is the best logits
            local_logits_best = []
            for epoch in range(args.E):
                logging.info('Epoch {}/{}: '.format(epoch + 1, args.E))
                train_loss_list = []
                logits_list = []
                for k in cur_selected_clients:
                    # ClientExecute()
                    client = clients[k]
                    logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))

                    """ 1. Local Training """
                    # θ_t^k ← ClientUpdate(D^k; θ_t-1^k)
                    train_loss = client.local_update()
                    train_loss_list.append(train_loss)
                    # Eval each client!
                    recorder.evaluate(client, validation_datasets[k - 1])
                    # Predict each client and record the loss of test as client's weight
                    if args.do_test:
                        recorder.predict(client, test_datasets[k - 1])

                    """ 2. Local Prediction """
                    # Compute local logits: Y_t^k ← f^k(X^0; θ_t^k)
                    logits = client.compute_logits(public_datasets)

                    logits_list.append(logits)

                # Get the best logits using minimum train loss
                index_logits = train_loss_list.index(min(train_loss_list))
                best_logit = logits_list[index_logits]

                """ 3. Upload """
                # Upload local best logits
                local_logits_best.append(best_logit)

            # ServerExecute()
            logger.info("# Node{:d}: {}_{}".format(server.id, server.name, server.model_type))
            """ 5. Server Distillation """
            for epoch in range(args.dis_epochs):
                logging.info('Distilling Epoch {}/{}:'.format(epoch + 1, args.dis_epochs))
                # ①server distillation: θ_t ← FedTLDistillation(D^0 ∪ {Y_t^1, Y_t^2, Y_t^3, Y_t^4, Y_t^5}; θ_t-1)
                server.fed_max_distillation(public_datasets, local_logits_best)  # aggregated by the server model
                # Eval server!
                recorder.evaluate(server, merged_val_datasets)
                # Predict server!
                if args.do_test:
                    recorder.predict(server, merged_test_datasets)
                    recorder.save_record(server, epoch, server.id, round_)
                    # The server predicts each test set separately and columns 6-10 are the values under each test set
                    for i in range(len(args.datasets) + 1, len(args.datasets) + 6):
                        server.id = i
                        recorder.predict(server, test_datasets[i - args.K - 1])
                        # Save record of server's distillation!
                        recorder.save_record(server, epoch, server.id, round_)
                server.id = 0
            # Save server!
            recorder.save_model(server)

            """ 6. Server Aggregation """
            # Compute server logits as aggregation: Y_t ← f(X^0; θ_t)
            logits_glob = server.compute_logits(public_datasets)

            """ 7. Local Distillation """
            # ②client distillation: θ_t^k ← ClientUpdate((X^0, Y_t); θ_t^k)
            logger.info("______ clients have received the logits_glob _____")
            for k in cur_selected_clients:
                # ClientExecute()
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))
                for epoch in range(args.dis_epochs):
                    logging.info('Distilling Epoch {}/{}:'.format(epoch + 1, args.dis_epochs))

                    client.ada_local_distillation(public_datasets, logits_glob)
                    # Eval each client!
                    recorder.evaluate(client, validation_datasets[k - 1])
                    # Predict each client!
                    if args.do_test:
                        recorder.predict(client, test_datasets[k - 1])

                    # Save record of client's distillation!
                    recorder.save_record(client, epoch, k, round_)
                # Save each client!
                recorder.save_model(client)

    elif args.algorithm == 'fed_tld':
    # elif args.algorithm == 'adafd_get_weights_by_gpt':
    # elif args.algorithm == 'fed_tld_inverse':
        # randomly sample partial clients: m = max(C*K, 1)
        m = max(int(args.C * args.K), 1)
        cur_selected_clients = sorted(np.random.choice(range(1, args.K + 1), m, replace=False))

        for round_ in range(args.R):
            logger.info('===============The {:d}-th round==============='.format(round_ + 1))

            logits_locals = []
            weights_with_train_loss = []
            query_gpt = ""
            for k in cur_selected_clients:
                # ClientExecute()
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))
                train_loss_list = []
                train_loss, f1 = 0., 0.
                for epoch in range(args.E):
                    logging.info('Epoch {}/{}: '.format(epoch + 1, args.E))

                    """ 1. Local Training """
                    # θ_t^k ← ClientUpdate(D^k; θ_t-1^k)
                    train_loss, f1 = client.local_update()
                    train_loss_list.append(train_loss)
                    # Eval each client!
                    recorder.evaluate(client, validation_datasets[k - 1])
                    # Predict each client and record the loss of test as client's weight
                    if args.do_test:
                        recorder.predict(client, test_datasets[k - 1])

                """ 2. Local Prediction """
                # Compute local logits: Y_t^k ← f^k(X^0; θ_t^k)
                logits = client.compute_logits(public_datasets)

                """ 3. Upload """
                # Upload local logits
                logits_locals.append(logits)

                # Let gpt analyze the weight distribution of the model (calculate the average value of each entropy)
                # entropy_logits = utils.compute_entropy(logits)
                # logger.info(f'entropy_logits: {entropy_logits}')
                # query_gpt += f"Client{k}: training loss={train_loss}, f1={f1}, logits={logits}, logits entropy={entropy_logits}\n"

                # choose the minimum train loss as client's weight
                weights_with_train_loss.append(min(train_loss_list))
            logger.info("the minimum train loss: {}".format(weights_with_train_loss))

            """ 4. compute weights  """
            # logger.info(f'query_gpt: {query_gpt}')
            # weights = get_weights_by_gpt(query_gpt)
            # logger.info(f"gpt's weights output: {weights}")
            # # print(f"before transforming type of gpt's weights output: {type(weights)}")
            # # Convert the output str to float type
            # weights = list(map(float, weights.split(',')))
            
            # loss v1: Calculate weights using inverse softmax
            alpha = 5
            weights = utils.calculate_softmax_inverse_proportions(weights_with_train_loss, alpha, cur_selected_clients, args.E)

            # loss v2: Calculate the weight by weighting the inverse of loss
            # weights = utils.calculate_linear_inverse_proportions(weights_with_train_loss)

            logger.info("Weights assigned to each client: {}".format(weights))
            logits_ensemble = server.logit_ensemble(logits_locals, weights)

            # ServerExecute()
            logger.info("# Node{:d}: {}_{}".format(server.id, server.name, server.model_type))
            """ 5. Server Distillation """
            for epoch in range(args.dis_epochs):
                logging.info('Distilling Epoch {}/{}:'.format(epoch + 1, args.dis_epochs))
                # ①server distillation: θ_t ← FedTLDistillation(D^0 ∪ {Y_t^1, Y_t^2, Y_t^3, Y_t^4, Y_t^5}; θ_t-1)
                # server.fed_tld_distillation(public_datasets, logits_locals, weights)  # aggregated by the server model
                server.fed_tld_distillation(public_datasets, logits_ensemble)  # aggregated by the server model
                # Eval server!
                recorder.evaluate(server, merged_val_datasets)
                # Predict server!
                if args.do_test:
                    recorder.predict(server, merged_test_datasets)
                    recorder.save_record(server, epoch, server.id, round_)
                    # The server predicts each test set separately and columns 6-10 are the values under each test set
                    for i in range(len(args.datasets) + 1, len(args.datasets) + 6):
                        server.id = i
                        recorder.predict(server, test_datasets[i - args.K - 1])
                        # Save record of server's distillation!
                        recorder.save_record(server, epoch, server.id, round_)
                server.id = 0
                
            # Save the server!
            recorder.save_model(server)

            """ 6. Server Aggregation """
            # Compute server logits as aggregation: Y_t ← f(X^0; θ_t)
            logits_glob = server.compute_logits(public_datasets)

            """ 7. Local Distillation """
            # ②client distillation: θ_t^k ← ClientUpdate((X^0, Y_t); θ_t^k)
            logger.info("______ clients have received the logits_glob _____")
            for k in cur_selected_clients:
                # ClientExecute()
                client = clients[k]
                logger.info("# Node{:d}: {}_{}".format(client.id, client.name, client.model_type))
                for epoch in range(args.dis_epochs):
                    logging.info('Distilling Epoch {}/{}:'.format(epoch + 1, args.dis_epochs))

                    client.ada_local_distillation(public_datasets, logits_glob)
                    # Eval each client!
                    recorder.evaluate(client, validation_datasets[k - 1])
                    # Predict each client!
                    if args.do_test:
                        recorder.predict(client, test_datasets[k - 1])

                    # Save record of client's distillation!
                    recorder.save_record(client, epoch, k, round_)
                # Save each client!
                recorder.save_model(client)