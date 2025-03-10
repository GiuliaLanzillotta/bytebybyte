import logging
import os
import sys

from evaluation import evaluate_agent_all_tasks_env, evaluate_agent_task

# Add project_dir to the Python path - necessary to solve relative imports
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_dir)

import numpy as np
import matplotlib.pyplot as plt
from environments import ClearWorld, MixedPermutationWorld, MultiDatasetsWorld, PermutationWorld, LabelShufflingWorld, get_environment_from_name
from agents import get_agent_class_from_name
from utils import *
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from datetime import datetime
import pprint
from hyperparameters import agent_hyperparameters
import argparse
import random
import string

def random_string(length=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

LOG_FILE = 'experiments_icml.json'
NUM_WORKERS = 8
EVAL_EVERY = 10 # evaluate every 10 training steps

parser = argparse.ArgumentParser(description="Input parameters from the command line.")
# COMMAND LINE ARGUMENTS 
# experiment setup arguments
parser.add_argument('--seed', type=int, required=False, default=np.random.randint(1,100), help='Random seed (integer)')
parser.add_argument('--device', type=int, required=True, help='Device identifier (integer).')
parser.add_argument('--steps_per_task', type=int, required=False, nargs='+', default=[3000], help='Sequence of update steps for each task')
parser.add_argument('--number_tasks', type=int, required=False, default=10, help='Number of the permutation tasks (integer)')
parser.add_argument('--environment', type=str, required=False, default="split", help='Name of the dataset (string)')
parser.add_argument('--exp_id', type=str, required=False, default=random_string(), help='Name of the experiment (string)')
parser.add_argument('--checkpoint_freq', type=int, required=False, default=0, help='Frequency of saving checkpoints (integer). if 0, no checkpoints are saved')
parser.add_argument('--wandb-project', type=str, required=True, help="name of wandb project")
# dataset-specific  arguments
parser.add_argument('--permutation_size', type=int, required=False, default=0, help='Size of the permutation (integer)')
parser.add_argument('--shuffling_fraction', type=float, required=False, default=0., help='Fraction of the labels shuffled (float)')
parser.add_argument('--random_order', action='store_true', help='Flag to use random ordering of tasks')
parser.add_argument('--task_sequence', type=int, required=False, nargs='+', help='Sequence of task indices (space-separated)')
parser.add_argument('--split_type', type=str, required=False, default="classes", choices=['chunks', 'classes'], help='Type of data split in split experiments (string)')
# agent-specific arguments 
parser.add_argument('--network_name', type=str, required=True, help='Name of the network used (string)')
parser.add_argument('--multihead', action='store_true', help='Flag to use multihead network')
parser.add_argument('--agent_type', type=str, required=True, choices=['base','regularization','replay'], help='Type of the agent (string)')

# Parse the arguments
args = parser.parse_args()
# Access the arguments
device = args.device
environment_name = args.environment
network_name = args.network_name
seed = args.seed
number_tasks = args.number_tasks
steps_per_task = args.steps_per_task
if len(steps_per_task)<number_tasks: 
    if len(steps_per_task)==1:
        steps_per_task = steps_per_task*number_tasks
    else: raise ValueError("The number of steps per task should be equal to the number of tasks or a single value")
agent_type = args.agent_type
checkpoint_freq = args.checkpoint_freq
split_type = args.split_type
multihead = args.multihead


# Set random seed for reproducibility
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Device setup 
device = device if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Multi-task environment
env, environment_name = get_environment_from_name(environment_name, args)
env_names = env.task_names # extracting task names
batches_eval = env.batches_eval
eval_criterion = env.criterion
num_classes_per_task = env.num_classes_per_task 
num_classes_total = env.num_classes
# taking care of task ordering ---
default_task_ordering = list(range(number_tasks))
if args.random_order: 
    ordering = np.random.shuffle(default_task_ordering)
else: 
    if args.task_sequence is not None: 
          assert len(args.task_sequence) == number_tasks, "The task ordering should include all tasks"
          ordering = args.task_sequence
    else: ordering = default_task_ordering
env.order_task_list(ordering)

# --- 

# configs setup 
experiment_name =  environment_name+"_"+network_name+"_"+agent_type+("_multihead" if multihead else "_singlehead")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
experiment_config = {
    "exp_id": args.exp_id,
    "exp_timestamp": timestamp,
    "exp_name": experiment_name,
    "seed": seed,
    "steps_per_task" : steps_per_task,
    "num_tasks" : len(env_names),
    "ordering" : env.ordered_task_names,
    "environment": env.world_name,
    "agent_type":agent_type, 
    "num_classes_per_task":num_classes_per_task,
    "num_classes_total":num_classes_total,
    "multihead": multihead
}

experiment_duration = sum(experiment_config['steps_per_task'])  # Total number of update steps
experiment_config['experiment_duration'] = experiment_duration


print(f"Experiment {experiment_name} started : {timestamp}")
pprint.pprint(experiment_config)


# Initialize agent
# Agent initialized and config files filled
if not experiment_name in agent_hyperparameters.keys(): 
    print(f"No stored hp for {experiment_name} experiment. Using default.")
hp_dict = agent_hyperparameters.get(experiment_name, {}) 
agent_class = get_agent_class_from_name(agent_type)
agent = agent_class(device,  **experiment_config, **hp_dict)
agent_config = agent.config # collect the filled config (with all the agent- and experiment-related info)

# Logger initialized
experiment_logger = ExperimentLogger(args.exp_id, timestamp, experiment_name, log_to_file=False, external_json_file=LOG_FILE, log_wandb=True, wandb_project=args.wandb_project, config=agent_config)

# Training loop
t = 0
for current_task, steps in enumerate(experiment_config['steps_per_task']):
    # before starting the task 
    current_env_name = env_names[current_task]
    logging.info(f"\n Training on task {current_env_name} ")

    # initializing task objective and training data iterator
    train_data = env.init_multi_task(number_of_tasks=current_task+1, train=True)
    #train_data = env.init_single_task(task_number=current_task, train=True)
    train_data_iterator = iter(DataLoader(train_data, batch_size=agent_config['batch_size'], shuffle=True, num_workers=NUM_WORKERS))
    # initialize training-time evaluation data 
    test_data = env.init_single_task(task_number=current_task, train=False) # same test data for both agents
    test_data_iterator = iter(DataLoader(test_data, batch_size=128, shuffle=True, num_workers=NUM_WORKERS)) 

    #reset agent optimizer before starting the new task 
    if current_task > 0: agent.reset_optimizer(task=current_task)

    for step in range(steps): 
        #checkpoint saving
        if checkpoint_freq>0 and t % checkpoint_freq == 0: 
            experiment_logger.save_checkpoint(agent, current_task)

        agent.ready_train() 
        try: train_loss, train_error = agent.update_one_step(train_data_iterator, current_task)
        except StopIteration: 
                # re-initialise train iterator
                train_data_iterator = iter(DataLoader(train_data, batch_size=agent_config['batch_size'], shuffle=True, num_workers=NUM_WORKERS))
                train_loss, train_error = agent.update_one_step(train_data_iterator, current_task)
        

        # logging and evaluation
        res = {}
        res['lr'] = agent.get_lr()
        res['train_loss']=train_loss
        res['train_error']=train_error

        if t % EVAL_EVERY == 0:
            try: 
                eval_res = evaluate_agent_task(batches_eval, agent, test_data_iterator, eval_criterion, ntasks_observed=current_task)
            except StopIteration: 
                # re-initialise test iterator
                test_data_iterator = iter(DataLoader(test_data, batch_size=128, shuffle=True, num_workers=NUM_WORKERS)) 
                eval_res = evaluate_agent_task(batches_eval, agent, test_data_iterator, eval_criterion, ntasks_observed=current_task)
            res.update(eval_res)
            experiment_logger.log(res, t)
        
        t+=1

    #TODO do end of task evaluations 
    # evaluate performance on all other tasks (forward and backward transfer)
    res_env = evaluate_agent_all_tasks_env(batches_eval, agent, env, train=False, ntasks_observed=current_task)
    experiment_logger.log_named_metrics(res_env, "transfer", current_task)

#checkpoint saving
if checkpoint_freq>0: 
    experiment_logger.save_checkpoint(agent, current_task)

logging.info("Training completed")

# do end of task evaluations 
res_env = evaluate_agent_all_tasks_env(batches_eval, agent, env, train=False, ntasks_observed=-1)
experiment_logger.log_named_metrics(res_env, "transfer", current_task)

#produce final metrics and log 
# offline evaluation 
all_data = env.init_multi_task(number_of_tasks=-1, train=False) # same test data for both agents
data_iterator = iter(DataLoader(all_data, batch_size=128, shuffle=True, num_workers=8)) 
# average environment evaluation
res = evaluate_agent_task(10, agent, data_iterator, eval_criterion, ntasks_observed=-1)
experiment_logger.close(res)
