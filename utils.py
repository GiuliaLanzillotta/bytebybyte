import wandb
import logging
import json
import os
from credentials import wandb_setup
import pandas as pd

import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import glob
import random
import string

from viz_utils import *

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def random_string(length=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def has_batch_norm(net):
    for n, p in net.named_modules():
        if isinstance(p, torch.nn.BatchNorm2d): return True
    return False

def get_norm_distance(m1, m2):
    return torch.linalg.norm(m1-m2, 2).item()

def get_cosine_similarity(m1, m2):
    cosine = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    return cosine(m1, m2)

def get_params(net):
    # Initialize an empty list to store the parameters
    params_list = []

    # Iterate over all the parameters of the network and append them to the list
    for param in net.parameters():
        params_list.append(param.data.view(-1))

    # Concatenate the list to a single tensor
    params_vector = torch.cat(params_list)

    return params_vector.detach().clone()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
 

def save_model_checkpoint(model, optimizer, path, **kwargs):
    """Save a model checkpoint to the specified path."""
    # Save checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    for k,v in kwargs.items():
        checkpoint[k] = v

    torch.save(checkpoint, path)

def load_model_checkpoint(model, optimizer, path, keys_to_load=[]):
    """Load a model checkpoint from the specified path."""
    # Load checkpoint
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    info = {}
    for k in keys_to_load:
        if k in checkpoint:
            info[k] = checkpoint[k] 
    config = checkpoint.get('config', None)
    return info, config


class AgentLogger:

    def __init__(self, agent_type, exp_id, config=None):
        """
        Initialize the logger with options for optional file logging (with timestamp), external JSON logging, and WandB.
        This logger is an Agent specific logger, which writes out the update for the given agent.
        """
        # Initialize timestamp for log file naming
        self.agent_type = agent_type
        self.exp_id = exp_id

        # Initialize cumulative metrics 
        self.timer = 0
        self.cumulative_error = 0;  self.cumulative_errors = []
        self.cumulative_train_error = 0;  self.cumulative__errors = []

        # Initialize the external JSON log file
        self.current_experiment = {'exp_id':exp_id, 'agent_type':agent_type, 'metrics': []} 

        # Log the configuration dictionary if provided
        self.log_config(config)

    def save_checkpoint(self, agent, directory, current_task, step):
        path = directory+f"{self.exp_id}-t{current_task}-s{step}.pth"
        logging.info(f"Saving model checkpoint to {path}")
        info = {'timer': self.timer}
        
        save_model_checkpoint(agent.network, agent.optimizer, path, **info, config=agent.config)

    @staticmethod
    def get_latest_checkpoint_path(directory, exp_id, task=None):
            checkpoint_paths = glob.glob(directory + f"{exp_id}-t{task if task is not None else '*'}-s*.pth")
            if not checkpoint_paths:
                return None
            latest_checkpoint = max(checkpoint_paths, key=os.path.getctime)
            return latest_checkpoint
    
    def load_checkpoint(self, agent, directory, task_to_load, step=None):
        if step is None: 
            path = self.get_latest_checkpoint_path(directory, self.exp_id, task=task_to_load)
        else: 
            path = directory + f"{self.exp_id}-t{task_to_load}-s{step}.pth"
        if path:
            logging.info(f"Loading model checkpoint from {path}")
            info, config = load_model_checkpoint(agent.network, agent.optimizer, path, ["timer"])
            self.timer = info['timer']
            agent.config = config
            return agent

        logging.warning(f"No checkpoints found in {directory} for experiment {self.exp_id}")

    def log_config(self, config):
        """Log the configuration dictionary to console, file, and WandB."""

        # Log config to the external JSON file
        self.current_experiment['config'] = config

        # Logging config to console and log file 
        config_message = f"Training Configuration -{self.agent_type} Agent: {config}"
        logging.info(config_message)


    def log(self, metrics, step=None):
        """
        Log metrics to the console, file, external JSON file, and WandB.
        
        Args:
            metrics (dict): Dictionary of metrics to log (e.g., {'loss': 0.5, 'accuracy': 0.9})
            step (int): Current step/epoch (optional, used for better tracking)
        """
        if step is None:
            self.timer +=1 
        else: self.timer=step 

        log_message = ', '.join([f'{key}: {value}' for key, value in metrics.items()])
        full_message = f'Step {self.timer} {self.agent_type} Agent: {log_message}' 

        # Log to console and file
        logging.info(full_message)

        # Log to WandB
        if wandb.run is not None:
            wandb.log(metrics, step=self.timer)

    
    def log_named_metrics(self, metrics, name, current_task):
        """
        Logging metrics with a specific name in the results dictionary
        """
        if current_task==0: self.current_experiment[name] = {}
        self.current_experiment[name][current_task] = metrics


    def close(self, end_results=None):
        """
        Close the logger. Finalize WandB and log summary statistics to the external JSON file.
        
        Args:
            summary_stats (dict): Dictionary of summary statistics (e.g., final loss, accuracy).
        """
        # Log summary stats if provided
        if end_results:
            logging.info(f"Final evaluation {self.agent_type} Agent: {end_results}")

            # Log to external JSON file
            self.current_experiment['end_results'] = end_results


        # Close WandB
        if wandb.run is not None:
            # we need to add the agent_name suffix
            wandb.log(end_results)
    


class ExperimentLogger:

    def __init__(self, exp_id, exp_timestamp, exp_name, log_to_file=False, external_json_file=None, log_wandb=False, wandb_project = None, config=None, log_outputs=False):
        """
        Initialize the logger with options for optional file logging (with timestamp), external JSON logging, and WandB.
        This logger tracks the experiment with two agents. 
        Args:
            supervised (bool): Whther the agent is supervised or RL
            log_to_file (bool): Whether to log to a file in the 'log/' directory.
            external_json_file (str): Path to an additional JSON log file (if needed).
            wandb_project_name (str): Name of the WandB project.
            config (dict): Configuration dictionary to log at the start.
        """
        # Initialize timestamp for log file naming
        self.exp_id = exp_id
        self.exp_name = exp_name
        agent_logger_class = AgentLogger 
        self.exp_directory = f"./experiments/{exp_name}-{exp_id}-{exp_timestamp}/"
        os.makedirs(self.exp_directory, exist_ok=True)
        self.log_outputs = log_outputs

        # Setup file logging if enabled
        if log_to_file:
            os.makedirs(f'{self.exp_directory}/logs', exist_ok=True)  # Create the log directory if it doesn't exist
            log_file = f'{self.exp_directory}/logs/training.log'
            logging.basicConfig(
                filename=log_file,
                filemode='a',
                format='%(asctime)s - %(levelname)s - %(message)s',
                level=logging.INFO
            )

            # Setup console logging
            self.console = logging.StreamHandler()
            self.console.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            logging.getLogger().addHandler(self.console)

        else:
            # Console-only logging
            logging.basicConfig(level=logging.INFO)
            self.console = logging.StreamHandler()
            self.console.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            logging.getLogger().addHandler(self.console)

        if log_outputs:
            # Setup file logging for outputs
            os.makedirs(f'{self.exp_directory}/logs', exist_ok=True)


        # Setup Agents logger  
        self.agent_logger = agent_logger_class(config['agent_type'], exp_id, config=config)

        # Initialize the external JSON log fi dle
        self.external_json_file = external_json_file

        # Setup WandB if a project name is provided
        if log_wandb:
            wandb.init(entity=wandb_setup['entity'], 
                       project=wandb_project, 
                       config=config)
            self.wandb_config = wandb.config
        
        

    def log(self, metrics, step=None):
        """
        Log metrics to the console, file, external JSON file, and WandB.
        
        Args:
            metrics (dict): Dictionary of metrics to log (e.g., {'loss': 0.5, 'accuracy': 0.9})
            step (int): Current step/epoch (optional, used for better tracking)
        """
        self.agent_logger.log(metrics, step)

    def log_named_metrics(self, metrics, name, current_task):
        """
        Logging transfer metrics (once at the end of each task)

        """
        self.agent_logger.log_named_metrics(metrics, name, current_task)

    def save_checkpoint(self, agent, current_task, step):
        directory = f'{self.exp_directory}/checkpoints/'
        os.makedirs(directory, exist_ok=True)
        self.agent_logger.save_checkpoint(agent, directory, current_task, step)

    def load_checkpoint(self, agent, task_to_load, step=None):
        directory = f'{self.exp_directory}/checkpoints/'
        agent = self.agent_logger.load_checkpoint(agent, directory, task_to_load, step)
        return agent

    def log_outputs(self, outputs, task_info,  current_task):
        """ Outputs is a tensor of size #num_steps x #num_samples x #out_dim. 
            task_info is a list with the task index for each sample."""
        filename = f'{self.exp_directory}/logs/outputs_task{current_task}.pt'
        print(f"Saving the outputs to ", filename)
        torch.save({
            "outputs": outputs,
            "task_info": task_info,
            "task_index": current_task,
        }, filename)

    def log_matrix(self, mat, name, plot=False, title=None, xaxis=None, yaxis=None): 
        """ name is not the full path, just the name of the matrix (without extension)"""
        # Ensure the results directory exists
        results_dir = f'{self.exp_directory}results/'
        os.makedirs(results_dir, exist_ok=True)

        # Convert torch tensor to numpy array if needed
        if isinstance(mat, torch.Tensor):
            mat = mat.cpu().numpy()

        # Save the numpy matrix to a file
        full_path = os.path.join(results_dir, f"{name}.npy")
        print(f"Saving the {name} matrix to ", full_path)
        np.save(full_path, mat)

        if plot: 
            self.plot_matrix(mat, title, xaxis, yaxis, name)

    def plot_matrix(self, mat, title, xaxis, yaxis, filename):
        """ filename is not the full path, just the name of the figure (without extension)"""
        results_dir = f'{self.exp_directory}results/'
        os.makedirs(results_dir, exist_ok=True)

        full_path = os.path.join(results_dir, f"{filename}.pdf")
        print(f"Saving the -{filename}- heatmap to ", full_path)
        
        # Convert torch tensor to numpy array if needed
        if isinstance(mat, torch.Tensor):
            mat = mat.cpu().numpy()

        viz_heatmap(mat, title, xaxis, yaxis, full_path)

    def plot_lines(self, mat, title, xaxis, yaxis, name, lbl_names=None, ylim=None):
        """ name is not the full path, just the name of the figure (without extension)"""
        results_dir = f'{self.exp_directory}results/'
        os.makedirs(results_dir, exist_ok=True)

        full_path = os.path.join(results_dir, f"{name}.pdf")
        print(f"Saving the -{name}- lineplot to ", full_path)

        # Convert torch tensor to numpy array if needed
        if isinstance(mat, torch.Tensor):
            mat = mat.cpu().numpy()

        viz_lineplots(mat, title, xaxis, yaxis, full_path, lbl_names=lbl_names, ylim=ylim)

    def close(self, end_results=None):
        """
        Close the logger. Finalize WandB and log summary statistics to the external JSON file.
        
        Args:
            summary_stats (dict): Dictionary of summary statistics (e.g., final loss, accuracy).
        """

        if end_results:  self.agent_logger.close(end_results)

        # Append current experiment to the external JSON file
        if self.external_json_file:
            exp_log = [self.agent_logger.current_experiment]
            path = f'{self.exp_directory}results/{self.external_json_file}'
            if not os.path.exists(path):
                # If file doesn't exist, start with a list
                os.makedirs(f'{self.exp_directory}results/', exist_ok=True)  # Create the results directory if it doesn't exist
                with open(path, 'w') as f:
                    json.dump(exp_log, f, indent=4)
            else:
                # Append to the existing JSON file
                with open(path, 'r+') as f:
                    try:
                        data = json.load(f)
                        data += exp_log
                        f.seek(0)
                        json.dump(data, f, indent=4)
                    except json.JSONDecodeError:
                        # Handle case where the file is empty or corrupted
                        json.dump(exp_log, f, indent=4)


        # Close WandB
        if wandb.run is not None:
            wandb.finish()

        logging.info("\n -* EXPERIMENT FINISHED. *-  \n")
        # Close and remove the logging handlers
        logging.shutdown()




def set_seed(seed):
    """ Function to set the random seed for reproducibility across libraries. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

