import torch
import torch.nn as nn

import lc
from lc.torch import ParameterTorch as Param, AsVector, AsIs
from lc.compression_types import ConstraintL0Pruning, ConstraintL1Pruning
from constants import Constants

from utils import evaluate_performance

class LC_Pruning:
    def __init__(self, args, trainloader, testloader):
        self.args = args
        self.trainloader = trainloader
        self.testloader = testloader
    
    def evaluate_performance(self, net):
        return evaluate_performance(net, self.testloader)
    
    def l_step(self, model, penalty, step):
        # train_loader, test_loader = x_train_tensor, x_test_tensor
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = torch.optim.SGD(params, lr=self.args["lr"])
        epochs_per_step_ = Constants.LC["epochs_per_step"]

        loss_fn = torch.nn.BCEWithLogitsLoss()

        for epoch in range(epochs_per_step_):
            model.train()
            for x, y in self.trainloader:
                optimizer.zero_grad()
                y_pred = model(x)
                loss = loss_fn(y_pred, y) + penalty()
                loss.backward()
                optimizer.step()
        return model
    
    def lc_prune(self, lc_model, perc_prune):
        lc_epochs_per_step = Constants.LC["epochs_per_step"]
        n_steps = self.args["num_epochs"] // lc_epochs_per_step + int(self.args["num_epochs"] % lc_epochs_per_step != 0)
        mu_s = [10 for i in range(n_steps)]
        layers = [lambda x=x: getattr(x, 'weight') for x in lc_model.modules() if isinstance(x, nn.Linear)]
        model_size = sum(p.numel() for p in lc_model.parameters() if p.requires_grad)
        # L0 pruning
        compression_tasks = {
            Param(layers, self.args["device"]): (AsVector, ConstraintL0Pruning(kappa=int(model_size * (1 - perc_prune))), 'pruning')
        }

        lc_alg = lc.Algorithm(
            model=lc_model,                            # model to compress
            compression_tasks=compression_tasks,  # specifications of compression
            l_step_optimization=self.l_step,        # implementation of L-step
            mu_schedule=mu_s,                     # schedule of mu values
            evaluation_func=self.evaluate_performance      # evaluation function
        )
        lc_alg.run()

        for name, parameter in lc_model.named_parameters():
            parameter.data[parameter.data.abs() < Constants.LC["prune_threshold"]] = 0
        perc_lc_pruned = self.fetch_perc_prune(lc_model)
        return lc_model, perc_lc_pruned
    
    def fetch_perc_prune(self, model):
        pruned = 0
        total = 0
        for name, parameter in model.named_parameters():
            pruned += torch.sum(parameter == 0).item()
            total += parameter.numel()
        return pruned/total