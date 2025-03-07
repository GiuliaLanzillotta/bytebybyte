""" Definition of supervised learning algorithms"""
import copy
import torch
import torchvision
import torch.nn as nn
import pprint 
import random
from datetime import date
import pytorch_warmup as warmup

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import get_network_from_name

    
class BaseAgent:
    """Simply fits the task by minimizing the objective.
        It wraps a network model, and an optimizer.
    """
    
    default_config = {
        "network":"resnet18",
        "optimizer":"SGD",
        "lr":0.1,
        "weight_decay":1e-5,
        "momentum":0.9,
        "step_scheduler_decay":300,
        "scheduler_step":0.1,
        "scheduler_type":"step", #cosine_anneal
        "loss":"CE",
        "batch_size":256,
        "warmup_on": True
    }

    default_losses = {
        "CE":nn.CrossEntropyLoss(),
        "MSE":nn.MSELoss()
    }

    default_optimizers = {
        "SGD": torch.optim.SGD,
        "Adam": torch.optim.Adam
    }

    def __init__(self, device, **kwargs) -> None:
        
        self.config = copy.deepcopy(self.default_config)
        self.config.update(kwargs) #substituting the values with new ones
        self.config['device'] = device
        self.config['date'] = str(date.today())
        self.dynamic_output = self.config["num_classes_per_task"] < self.config["num_classes_total"]
        if self.dynamic_output: print("Dynamic output mode.")
        self.setup()
        print("Setup complete.")

    def change_config(self, new_config):
        self.config.update(new_config)

    def setup(self):
        """Initializing network, loss, optimizer, scheduler."""

        self.network = get_network_from_name(self.config['network'], **self.config)
        self.network.cuda(self.config['device'])
        self.loss = self.default_losses[self.config['loss']]
        self.setup_optimizer()
        self.setup_scheduler()


    def setup_optimizer(self):
        """ Initializing optimizer"""
        self.optimizer = \
            self.default_optimizers[self.config['optimizer']](self.network.parameters(), lr = self.config['lr'], weight_decay=self.config['weight_decay'],momentum=self.config['momentum'])

    def setup_scheduler(self, task=0):
        """ Initializing scheduler 
            - restart flag: whether reinitializing for a new task
        """
        if self.config['scheduler_type']=="step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(self.config['step_scheduler_decay']), gamma=self.config['scheduler_step'])
        elif self.config['scheduler_type'] == "cosine_anneal":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=self.config['steps_per_task'][task], T_mult=1, eta_min=1e-5)
        else: 
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1, total_iters=-1)
        if self.config["warmup_on"]:
            self.warmup_scheduler = warmup.LinearWarmup(self.optimizer, warmup_period=self.config['steps_per_task'][task]//100)   

    def reset_optimizer(self, task=0):
        self.setup_optimizer()
        self.setup_scheduler(task)         

    # Function to get current learning rate from the optimizer
    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        
    def update(self, loss):
        """ performs the parameter update on the network"""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.config["warmup_on"]:
            with self.warmup_scheduler.dampening(): 
                self.scheduler.step()
        else: self.scheduler.step()

    def compute_dynamic_output(self, output, current_task, input_task_index):
        """ Artificially removing some classes from prediction. """
        nct = self.config["num_classes_per_task"]
        if self.config["multihead"]:
            if not torch.all(input_task_index == input_task_index[0]):
                raise AssertionError("Not all elements in the vector have the same value.")
            start_index = nct*input_task_index
            end_index = nct*(input_task_index+1)
            for i in range(len(start_index)):
                output[i,:start_index[i]].data.fill_(-10e10)
                output[i,end_index[i]:].data.fill_(-10e10)   
            
        else: 
            end_index = nct*(current_task+1)
            output[:,end_index:].data.fill_(-10e10)
        return output

    def compute_loss(self, data_iterator, current_task=-1):
        """Samples one batch from the data iterator and computes the loss.
        Returns accuracy and loss.
        """
        x, y, task_id = next(data_iterator) 
        device = f"cuda:{self.config['device']}"
        x = x.to(device); y = y.to(device)
        output = self.network(x)
        if self.dynamic_output: 
            # exclude classes from the loss 
            output = self.compute_dynamic_output(output, current_task, task_id)
        loss = self.loss(output, y)
        acc = torch.sum((torch.max(output,dim=1)[1] == y).float())/len(y)
        return loss, acc

    def update_one_step(self, data_iterator, current_task=-1):
        loss, acc = self.compute_loss(data_iterator, current_task)
        self.update(loss)

        return loss.detach().item(), 1 - acc.detach().item()

    def ready_eval(self):
        """Puts the network in evaluation mode"""
        self.network.eval()

    def ready_train(self):
        """Puts the network in train mode"""
        self.network.train()

    def predict(self, inputs, current_task=-1):
        x, y, task_id = inputs
        x = x.to(self.config['device'])
        y = y.to(self.config['device'])
        out =  self.network(x)
        if self.dynamic_output: 
            # exclude classes from the loss 
            out = self.compute_dynamic_output(out, current_task, task_id)
        return out,y
    
    def reset(self):
        self.setup()


class RegularizationAgent(BaseAgent):
    """Uses regularization on top of the current task objective.
    """
    
    default_config = { #TODO: make a separate config for regularization
        "network":"resnet18",
        "optimizer":"SGD",
        "lr":0.1,
        "weight_decay":1e-5,
        "momentum":0.9,
        "step_scheduler_decay":300,
        "scheduler_step":0.1,
        "scheduler_type":"step", #cosine_anneal
        "loss":"CE",
        "batch_size":256,
        "warmup_on": True,
        "regularization_strength": 0.01,
        "regularizer":"EWC"
    }

    def __init__(self, device, **kwargs) -> None:
        super().__init__(device=device,**kwargs)
        self.regularizer_name = self.config['regularizer']
         # regularization is on
        
    def compute_ewc(self):
        """Computes the EWC regularization term on the network parameters.
        Returns the regularization term of the loss."""
        #TODO
        return 0

    def compute_regularization(self, data_iterator=None):
        """ Computes a regularization on the network parameters.
        Returns the regularization term of the loss."""
        if self.regularizer_name == "EWC":
            return self.compute_ewc()
        else:
            raise NotImplementedError

    def update_one_step(self, data_iterator, current_task=-1):
        """ Performs one update step of the parameters based on the agent loss. It draws a batch from the data_iterator and the buffer_iterator. Note that the batches might be of different sizes, thus the replay is not necessarily balanced."""

        unregularized_loss, current_task_guesses = super().compute_loss(data_iterator, current_task)
        regularization = self.compute_regularization()
        loss = unregularized_loss + self.config['regularization_strength']*regularization
        self.update(loss)


        acc = (torch.sum(current_task_guesses.float()))/(len(current_task_guesses))

        #TODO: we should return the regularization and unregularized loss as well
        return loss.detach().item(), 1 - acc.detach().item()



class ReplayAgent(RegularizationAgent):
    """Uses simple experience replay on top of the current task objective (potentially regularised)
    """
    
    default_config = { #TODO: make a separate config for replay
        "network":"resnet18",
        "optimizer":"SGD",
        "lr":0.1,
        "weight_decay":1e-5,
        "momentum":0.9,
        "step_scheduler_decay":300,
        "scheduler_step":0.1,
        "scheduler_type":"step", #cosine_anneal
        "loss":"CE",
        "batch_size":256,
        "warmup_on": True,
        "regularization_strength": 0.01,
        "regularizer":"EWC",
        "buffer_size":500
    }

    def __init__(self, device, **kwargs) -> None:
        super().__init__(device=device,**kwargs)
        self.regularization_on = self.config['regularization_strength']>0
        

    def update_one_step(self, data_iterator, buffer_iterator=None, current_task=-1):
        """ Performs one update step of the parameters based on the agent loss. It draws a batch from the data_iterator and the buffer_iterator. Note that the batches might be of different sizes, thus the replay is not necessarily balanced."""

        if buffer_iterator is None: #simple update
            return super().update_one_step(data_iterator, current_task)
        
        # replay update 
        # we assume that the same loss is applied to the current task and buffer samples. This holds true only for ER 
        current_task_loss, current_task_guesses = super().compute_loss(data_iterator, current_task)
        buffer_loss, buffer_guesses = super().compute_loss(buffer_iterator, current_task)
        loss = current_task_loss + buffer_loss
        if self.regularization_on:
            regularization = self.compute_regularization()
            loss += self.config['regularization_strength']*regularization
        self.update(loss)

        acc = (torch.sum(current_task_guesses.float())+torch.sum(buffer_guesses.float()))/(len(current_task_guesses)+len(buffer_guesses))
        #TODO: return the loss and accuracy on both current task and buffer
        acc_current_task = (torch.sum(current_task_guesses.float()))/(len(current_task_guesses))
        acc_buffer = (torch.sum(buffer_guesses.float()))/(len(buffer_guesses))

        return loss.detach().item(), 1 - acc.detach().item()


def get_agent_class_from_name(agent_name):
    if agent_name=="base": return BaseAgent
    if agent_name=="regularization": return RegularizationAgent
    if agent_name=="replay": return ReplayAgent
