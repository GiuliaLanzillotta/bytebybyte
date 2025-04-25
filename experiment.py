import logging
import os
import sys


# Add project_dir to the Python path - necessary to solve relative imports
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_dir)

import numpy as np
import matplotlib.pyplot as plt
from environments import get_environment_from_name
from agents import get_agent_class_from_name
from utils import *
from evaluation import *
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from datetime import datetime
import pprint
from hyperparameters import agent_hyperparameters
import argparse

LOG_FILE = 'experiments.json'
NUM_WORKERS = 8
EVAL_EVERY = 10 # evaluate every 10 training steps
NUM_SAMPLES_PER_TASK = 1000 # number of samples per task for the function tracker

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
parser.add_argument('--experiment_type', type=str, required=True, choices=['multitask', 'singletask'], help='Type of the experiment (string)')
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
parser.add_argument('--replay_fraction', type=float, required=False, default=0.0, help='Fraction of replay data for each task (float)')
parser.add_argument('--replay_type', type=str, required=False, choices=['balanced', 'fixed'], default='balanced', help='Type of replay (string)')
parser.add_argument('--regularization_strength', type=float, required=False, default=0.0, help='Strength of the regularization (float)')
parser.add_argument('--regularizer', type=str, required=False, choices=['Null', 'EWC'], default='Null', help='Type of regularization (string)')
parser.add_argument('--batch_size', type=int, required=False, help='Batch size for training (integer)')
parser.add_argument('--num_replay_tasks', type=int, required=False,  default=-1, help='Number of tasks to replay (integer)')
parser.add_argument('--track_outputs', action='store_true', help='Flag to track and log model outputs during training')

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
exp_type = args.experiment_type
track_outputs = args.track_outputs


# Set random seed for reproducibility
seed_everything(seed)

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
experiment_name =  environment_name+"_"+network_name+"_"+agent_type+("_multihead" if multihead else "_singlehead")+"_"+exp_type
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
experiment_config = {
    "exp_id": args.exp_id,
    "exp_timestamp": timestamp,
    "exp_name": experiment_name,
    "exp_type": exp_type,
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
if not env.world_name in agent_hyperparameters.keys(): 
    print(f"No stored hp for {env.world_name} experiment. Using default.")
hp_dict = agent_hyperparameters.get(env.world_name, {})
# Default hyperparameters are overwritten by cdommand line args
for k,v in hp_dict.items():
     if k in vars(args).keys(): 
          new_v = vars(args)[k]
          if new_v is not None:
            print(f"Updating {k} from {v} tp {new_v}")
            hp_dict[k] = new_v
agent_class, agent_specific_args = get_agent_class_from_name(agent_type)
agent_specific_config = {key: vars(args).get(key, None) for key in agent_specific_args}
agent = agent_class(device,  **experiment_config, **hp_dict, **agent_specific_config)
agent_config = agent.config # collect the filled config (with all the agent- and experiment-related info)
replay_on = "replay" in agent_type
if replay_on: assert exp_type != "multitask", "Replay can only be used in a singletask setting"

# Logger initialized
experiment_logger = ExperimentLogger(args.exp_id, timestamp, experiment_name, log_to_file=False, external_json_file=LOG_FILE, log_wandb=True, wandb_project=args.wandb_project, config=agent_config, log_outputs=track_outputs)

# Training loop
t = 0
parameters_path = [get_params(agent.network)] # add initialization
if checkpoint_freq>0: experiment_logger.save_checkpoint(agent, "init", 0) # save initialization
if track_outputs: tracker = FunctionTracker(NUM_SAMPLES_PER_TASK, device, experiment_logger)

for current_task, steps in enumerate(experiment_config['steps_per_task']):
    # before starting the task 
    current_env_name = env_names[current_task]
    logging.info(f"\n Training on task {current_env_name} ")

    # initializing task objective and training data iterator
    batch_size=agent_config['batch_size']
    if exp_type=="multitask": 
        train_data = env.init_multi_task(number_of_tasks=current_task+1, train=True)
    else: train_data = env.init_single_task(task_number=current_task, train=True)
    if replay_on and current_task>0: 
        # adjust the batch size and initialise the buffer data
        batch_size, batch_size_replay, total_replay_tasks = agent.calculate_task_batchsize(current_task) # balanced replay,every task gets the same amount of data in 
        initial_task = current_task-total_replay_tasks
        print(f"Using batch sizes: {batch_size} (new) and {batch_size_replay} (old) for {total_replay_tasks} tasks")
        buffer_data = env.init_buffer((initial_task,current_task), buffer_size=agent_config['replay_fraction'])
        buffer_dataloader = DataLoader(buffer_data, batch_size=batch_size_replay*(total_replay_tasks), shuffle=True)
        buffer_data_iterator = iter(buffer_dataloader)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    train_data_iterator = iter(train_dataloader)
    # initialize training-time evaluation data 
    test_data = env.init_single_task(task_number=current_task, train=False) # same test data for both agents
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=True, num_workers=NUM_WORKERS)
    test_data_iterator = iter(test_dataloader) 

    #reset agent optimizer before starting the new task 
    if current_task > 0: agent.reset_optimizer(task=current_task)

    if track_outputs:
        tracker.add_inputs(iter(train_dataloader))

    for step in range(steps): 

        agent.ready_train() 
        # current task data
        try: data_point = next(train_data_iterator) 
        except StopIteration: 
                # re-initialise train iterator
                train_data_iterator = iter(train_dataloader)
                data_point = next(train_data_iterator) 
        if not replay_on or current_task==0: 
                train_loss, train_error = agent.update_one_step(data_point, current_task= current_task)
        else: # add replay 
            try: buffer_data_point = next(buffer_data_iterator)
            except StopIteration: 
                buffer_data_iterator = iter(buffer_dataloader)
                buffer_data_point = next(buffer_data_iterator)
            train_loss, train_error = agent.update_one_step(data_point, buffer_data_point, current_task)

        if track_outputs: 
            tracker.compute_outputs(agent)
        
    

        # logging and evaluation
        res = {}
        res['lr'] = agent.get_lr()
        res['train_loss']=train_loss
        res['train_error']=train_error

        if step % EVAL_EVERY == 0:
            try: 
                eval_res = evaluate_agent_task(batches_eval, agent, test_data_iterator, eval_criterion, ntasks_observed=current_task)
            except StopIteration: 
                # re-initialise test iterator
                test_data_iterator = iter(test_dataloader) 
                eval_res = evaluate_agent_task(batches_eval, agent, test_data_iterator, eval_criterion, ntasks_observed=current_task)
            res.update(eval_res)
            experiment_logger.log(res, t)
        
        # saving checkpoint 
        if checkpoint_freq>0 and (step+1) % checkpoint_freq == 0: 
            print(f"Saving checkpoint {step+1}!")
            experiment_logger.save_checkpoint(agent, current_task, step+1)

        t+=1
        
    if track_outputs:
        tracker.log_outputs()
    # do end of task evaluations 
    end_of_task_parameters = get_params(agent.network)
    parameters_path.append(end_of_task_parameters)
    # evaluate performance on all other tasks (forward and backward transfer)
    res_env = evaluate_agent_all_tasks_env(batches_eval, agent, env, train=False, ntasks_observed=current_task)
    experiment_logger.log_named_metrics(res_env, "transfer", current_task)


logging.info("Training completed")
#produce final metrics and log 
all_data = env.init_multi_task(number_of_tasks=-1, train=False) # same test data for both agents
multi_task = DataLoader(all_data, batch_size=128, shuffle=True, num_workers=8) 
# average/offline evaluation 
data_iterator = iter(multi_task)
# average environment evaluation
res = evaluate_agent_task(10, agent, data_iterator, eval_criterion, ntasks_observed=-1)
# renaming the keys before logging
final_res = {}
for k,v in res.items():
    final_res[f'{k}_end'] = v
# # ------------------------------------------------------------

experiment_logger.close(final_res)
