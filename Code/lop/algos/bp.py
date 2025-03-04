import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
from copy import deepcopy
import math
class Backprop(object):
    def __init__(self, net, step_size=0.001, loss='nll', opt='sgd', beta_1=0.9, beta_2=0.999, weight_decay=0.0,
                 to_perturb=False, perturb_scale=0.1, device='cpu', momentum=0):
        self.net = net
        self.to_perturb = to_perturb
        self.device = device

        # define the optimizer
        if opt == 'sgd':
            self.opt = optim.SGD(self.net.parameters(), lr=step_size, weight_decay=weight_decay, momentum=momentum)
        elif opt == 'adam':
            self.opt = optim.Adam(self.net.parameters(), lr=step_size, betas=(beta_1, beta_2),weight_decay=weight_decay)
        elif opt == 'adamW':
            self.opt = optim.AdamW(self.net.parameters(), lr=step_size, betas=(beta_1, beta_2),weight_decay=weight_decay)

        # define the loss function
        self.loss = loss
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[self.loss]


    def learn(self, x, target):
        for name, param in self.net.named_parameters():
            print(name, param.shape)
        self.opt.zero_grad()
        output, features = self.net.predict(x=x)
        loss = self.loss_func(output, target.long())

        loss.backward()
        self.opt.step()
        return loss.detach(), output.detach()
class decreaseBackprop(object):
    def __init__(self, net, step_size=0.001, loss='nll', opt='sgd', beta_1=0.9, beta_2=0.999, weight_decay=0.0,to_perturb=False, perturb_scale=0.1, device='cpu', momentum=0):
        self.net = net
        self.device = device

        # define the optimizer
        if opt == 'sgd':
            self.opt = optim.SGD(self.net.parameters(), lr=step_size, weight_decay=weight_decay, momentum=momentum)
        elif opt == 'adam':
            self.opt = optim.Adam(self.net.parameters(), lr=step_size, betas=(beta_1, beta_2),weight_decay=weight_decay)
        elif opt == 'adamW':
            self.opt = optim.AdamW(self.net.parameters(), lr=step_size, betas=(beta_1, beta_2),weight_decay=weight_decay)

        # define the loss function
        self.loss = loss
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[self.loss]


    def learn(self, x, target,task):
        layer_scaling = {
            "conv1.weight": 0.5,
            "conv1.bias": 0.5,
            "conv2.weight": 0.6,
            "conv2.bias": 0.6,
            "conv3.weight": 0.7,
            "conv3.bias": 0.7,
            "fc1.weight": 0.8,
            "fc1.bias": 0.8,
            "fc2.weight": 0.9,
            "fc2.bias": 0.9,
            "fc3.weight": 1.0,
            "fc3.bias": 1.0
        }
        layer_scaling = {name: scale + (1 - scale)*1.005**-task for name, scale in layer_scaling.items()}


        self.opt.zero_grad()
        output, features = self.net.predict(x=x)
        loss = self.loss_func(output, target.long())

        loss.backward()
        for name, param in self.net.named_parameters():
            if name in layer_scaling:
                param.grad *= layer_scaling[name]
        self.opt.step()
        return loss.detach(), output.detach()
class EWC_Policy(object):
    def __init__(self, net, step_size=0.001, loss='nll', weight_decay=0.0,opt="s",to_perturb=False, perturb_scale=0.1, device='cpu', momentum=0, lambda_ewc=1):
        self.net = net.to(device)
        self.device = device
        self.lambda_ewc = lambda_ewc  # Regularization strength for EWC
        self.opt = optim.SGD(self.net.parameters(), lr=step_size, weight_decay=weight_decay, momentum=momentum)
        self.loss = loss
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[self.loss]

        # Placeholder for Fisher matrix and old parameters
        self.fisher_matrix = None
        self.params_old = {n: p.clone().detach() for n, p in self.net.named_parameters() if p.requires_grad}
        self.ewc_penalty = 0.0

    def update_ewc_penalty(self, dataset):
        fisher = {}
        for n, p in self.net.named_parameters():
            if p.requires_grad:
                fisher[n] = torch.zeros_like(p, device=self.device)

        x_test, y_test,_,_ = dataset
        x_test, y_test = x_test.float().to(self.device), y_test.to(self.device)

        batch_size = 200  # Smaller batch for stability
        num_batches = len(x_test) // batch_size

        for i in range(num_batches):
            test_batch_x = x_test[i * batch_size: (i + 1) * batch_size]
            test_batch_y = y_test[i * batch_size: (i + 1) * batch_size]

            self.net.zero_grad()
            output, _ = self.net.predict(test_batch_x)
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), test_batch_y.long())
            negloglikelihood.backward()  # No need for retain_graph=True

            for n, p in self.net.named_parameters():
                if p.grad is not None:
                    fisher[n] += (p.grad.data ** 2) / num_batches
        self.fisher_matrix=fisher
        # Compute initial EWC penalty
        loss = 0.0
        for n, p in self.net.named_parameters():
            if n in self.fisher_matrix:
                loss += (self.fisher_matrix[n] * (p - self.params_old[n]) ** 2).sum()
        self.ewc_penalty = loss
        self.params_old = {n: p.clone().detach() for n, p in self.net.named_parameters() if p.requires_grad}


    def learn(self, x, target):
        self.opt.zero_grad()
        output, _ = self.net.predict(x.to(self.device))
        loss = self.loss_func(output, target.long().to(self.device)) + self.lambda_ewc * self.ewc_penalty
        loss.backward(retain_graph=True)
        self.opt.step()
        return loss.detach(), output.detach()



