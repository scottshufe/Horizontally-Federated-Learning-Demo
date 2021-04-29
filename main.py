import argparse, json
import datetime
import os
import logging
import torch, random

from server import *
from client import *
import models, datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()

    # Load config file
    with open(args.conf, 'r') as f:
    # with open('./utils/config.json', 'r') as f:
        conf = json.load(f)

    train_datasets, eval_datasets = datasets.get_dataset("./data/", conf['type'])

    # Create a server
    server = Server(conf, eval_datasets)
    clients = []

    # Create several clients
    for c in range(conf["no_models"]):
        clients.append(Client(conf, server.global_model, train_datasets, c))

    print("\n\n")

    for e in range(conf["global_epochs"]):
        # Sample k Clients to participate in federated learning in each epoch
        candidates = random.sample(clients, conf["k"])
        weight_accumulator = {}

        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        for c in candidates:
            diff = c.local_train(server.global_model)
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])

        # Model aggregation
        server.model_aggregate(weight_accumulator)

        # Evaluate global model
        acc, loss = server.model_eval()

        print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
