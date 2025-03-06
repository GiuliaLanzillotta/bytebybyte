import os
import sys

# Add project_dir to the Python path - necessary to solve relative imports
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_dir)

import numpy as np
import matplotlib.pyplot as plt
from supervised_experiments.supervised_envs import ClearWorld, MixedPermutationWorld, MultiDatasetsWorld, PermutationWorld
from supervised_experiments.supervised_agents import ReplayAgent, SGDAgent
from utils import *
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from datetime import datetime
import pprint
from supervised_experiments.old_agent_setups import *
import argparse

parser = argparse.ArgumentParser(description="Input parameters from the command line.")

# Add arguments
parser.add_argument('--seed', type=int, required=False, default=np.random.randint(1,100), help='Random seed (integer)')
parser.add_argument('--device', type=int, required=True, help='Device identifier (integer).')
parser.add_argument('--task_duration', type=int, required=False, default=3000, help='Duration of the task (integer)')
parser.add_argument('--number_tasks', type=int, required=False, default=10, help='Number of the permutation tasks (integer)')
parser.add_argument('--dataset', type=str, required=False, default="clear", help='Name of the dataset (string)')
parser.add_argument('--agent_name', type=str, required=False, default="mixed", help='Name of the agent type, e.g., fbmt, st, mixed (default) (string)')
parser.add_argument('--buffer_size', type=int, required=False, default=None, help='Buffer size (integer)')
parser.add_argument('--num_switches', type=int, required=False, default=1, help='Number of tasks switches (int)')

# Parse the arguments
args = parser.parse_args()
# Access the arguments
device = args.device
task_duration = args.task_duration
seed = args.seed
agent_name = args.agent_name
switches = args.num_switches 

# Multi-task environment
env = MixedPermutationWorld(10, num_switches=switches)
env_names = env.task_names # extracting task names
print(env_names)
print(env.switch_task_indices)
eval_criterion = nn.CrossEntropyLoss()
ordering = (False,None) #(to order?, order)

if ordering[0]: env.order_task_list(ordering[1])

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
experiment_config = {
    "exp_id": timestamp,
    "seed": seed,
    "task_duration" : task_duration,
    "num_tasks" : len(env_names),
    "ordering" : env.ordered_task_names,
    "data": env.world_name,
    "agent_name": agent_name
}


print(f"Experiment started : {timestamp}")
pprint.pprint(experiment_config)

# Set random seed for reproducibility
seed = experiment_config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

T = experiment_config['task_duration']*experiment_config['num_tasks']  # Total number of time steps

def evaluate_transfer(num_batches, agent, env, task_number):
    print("Done training. Evaluating transfer now.")
    res_offline = {}
    agent.ready_eval()

    # Evaluate on all the tasks 
    for i in range(env.number_tasks):
        test_data = env.init_single_task(task_number=task_number, train=False) # same test data for both agents
        test_data_iterator = iter(DataLoader(test_data, batch_size=128, shuffle=True, num_workers=4)) 
        res = evaluate_agent_env(num_batches, agent, test_data_iterator)
        
        res_offline[env.task_names[i]] = res
    
    return res_offline

def evaluate_agent_env(num_batches, agent, test_iter):
    # average loss and accuracy on 'num_batches' batches
    agent.ready_eval()

    loss = 0; acc=0; total=0
    for b in range(num_batches):
        x, y = next(test_iter)
        total+=len(y)
        with torch.no_grad():
            out_st,y = agent.predict((x,y))
            loss += eval_criterion(out_st, y)
            acc += torch.sum((torch.max(out_st,dim=1)[1] == y).float())

    loss /=(b+1)
    acc /= total

    results = {
            "test_loss":loss.item(),
            "test_error":(1 - acc.item())
        }
    return results

# Initialize agents 
device = device if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


agent_nature = "fbmt" if agent_name=="mixed" else agent_name
current_task_name = 'permuted-16'
_agent_config = agent_config[current_task_name][agent_nature]
if args.buffer_size is not None:
    _agent_config['buffer_size'] = args.buffer_size

# Agents initialized and config files filled
agent = ReplayAgent(device, 
                    **experiment_config,
                    **_agent_config) #start with MT objective
_agent_config = agent.config


# Logger initialized
experiment_logger = ExperimentLogger(timestamp, log_to_file=False, external_json_file='experiments_test.json', 
                                     log_wandb=True, configs=(_agent_config, experiment_config))

# Training loop
start_memory=0 # indexed of the first task in memory
for t in range(T):
    current_task = t // experiment_config['task_duration']
    if t % experiment_config['task_duration'] == 0: # if at the beginning of the task
        buffer_data_iterator = None
        if t>0: # at least one task training completed
             # evaluate transfer on all other datasets 
             transfer = evaluate_transfer(5, agent, env, current_task-1)
             experiment_logger.log_transfer(transfer, current_task-1)
        

        current_env_name = env_names[current_task]
        logging.info(f"\n Training on task {current_env_name} ")

        # initialise all the agents objectives 
        if current_task in env.switch_task_indices: 
            print("Switching strategy")
            if current_task_name == 'permuted-16': 
                current_task_name = 'permuted-32'
                if agent_name=="mixed": agent_nature = "st" 
            else: 
                current_task_name = 'permuted-16'
                if agent_name=="mixed":
                    agent_nature = "fbmt" 
                    start_memory = current_task
                

            agent.change_config(agent_config[current_task_name][agent_nature])
            _agent_config = agent.config
        
        if agent_nature=="fbmt": batch_size_task = _agent_config['batch_size']//(current_task-start_memory+1) 
        else: batch_size_task = _agent_config['batch_size']
        
        if current_task-start_memory > 0 and agent_nature == 'fbmt': 
                 buffer_data, buffer_indices = env.init_buffer((start_memory,current_task), agent.buffer_indices, buffer_size=_agent_config['buffer_size'])
                 agent.buffer_indices = buffer_indices
                 buffer_data_iterator = iter(DataLoader(buffer_data, batch_size=batch_size_task*(current_task-start_memory), shuffle=True))

        train_data = env.init_single_task(task_number=current_task, train=True)
        train_data_iterator = iter(DataLoader(train_data, batch_size=batch_size_task, shuffle=True))

        # initialize evaluation data 
        test_data = env.init_single_task(task_number=current_task, train=False) # same test data for both agents
        test_data_iterator = iter(DataLoader(test_data, batch_size=128, shuffle=True)) 

        #reset agents before starting the new task 
        if agent_nature=="st": agent.reset()
        if t>0: agent.reset_optimizer()
    
    agent.ready_train()

    try: train_loss, train_error = agent.update_one_step(train_data_iterator, buffer_data_iterator)
    except StopIteration: 
            # re-initialise train iterator
            train_data_iterator = iter(DataLoader(train_data, batch_size=batch_size_task, shuffle=True))
            if agent_nature == 'fbmt' and current_task-start_memory>0: 
                 buffer_data_iterator = iter(DataLoader(buffer_data, batch_size=batch_size_task*(current_task-start_memory), shuffle=True))
            train_loss, train_error = agent.update_one_step(train_data_iterator, buffer_data_iterator)

    try: res = evaluate_agent_env(1, agent, test_data_iterator)
    except StopIteration: 
        # re-initialise test iterator
        test_data_iterator = iter(DataLoader(test_data, batch_size=128, shuffle=True, num_workers=4)) 
        res = evaluate_agent_env(1, agent, test_data_iterator)


    res['lr'] = agent.get_lr()
    res['train_loss']=train_loss
    res['train_error']=train_error

    experiment_logger.log(res, t)

#produce final metrics and log 
# transfer evaluation 
transfer = evaluate_transfer(5, agent, env, current_task)
experiment_logger.log_transfer(transfer, current_task)
# offline evaluation 
offline_data = env.init_multi_task(number_of_tasks=-1, train=False) # same test data for both agents
offline_data_iterator = iter(DataLoader(offline_data, batch_size=128, shuffle=True)) 
res = evaluate_agent_env(10, agent, offline_data_iterator)
cum_errors = experiment_logger.close(res)
