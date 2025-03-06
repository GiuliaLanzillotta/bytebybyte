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

def read_results():
    # Load the JSON file
    file_path = './results/experiments.json'
    with open(file_path, 'r') as f:
        experiments_data = json.load(f)
    # Extract relevant information into a pandas DataFrame
    # We are assuming that the JSON file contains a list of experiments
    data = []

    for experiment in experiments_data:
        exp_id = experiment.get('exp_id', '')
        agent_name = experiment.get('agent_name', '')
        config = experiment.get('config', {})
        metrics = experiment.get('metrics', [])
        summary_stats = experiment.get('summary_stats', {})

        # Combine all relevant details into a flat dictionary
        row = {
            'exp_id': exp_id,
            'agent_name': agent_name,
            'network': config.get('network', ''),
            'optimizer': config.get('optimizer', ''),
            'lr': config.get('lr', 0),
            'weight_decay': config.get('weight_decay', 0),
            'momentum': config.get('momentum', 0),
            'step_scheduler_decay': config.get('step_scheduler_decay', 0),
            'scheduler_step': config.get('scheduler_step', 0),
            'scheduler_type': config.get('scheduler_type', ''),
            'loss': config.get('loss', ''),
            'batch_size': config.get('batch_size', 0),
            'exp_seed': config.get('seed', 0),
            'task_duration': config.get('task_duration', 0),
            'num_tasks': config.get('num_tasks', 0),
            'data': config.get('data', ''),
            'device': config.get('device', 0),
            'test_loss_offline': summary_stats.get('test_loss_offline', 0),
            'test_error_offline': summary_stats.get('test_error_offline', 0),
            'cumulative_error': summary_stats.get('cumulative_error', 0),
            'average_error': summary_stats.get('average_error', 0)
        }
        
        data.append(row)

    # Create a pandas DataFrame
    df = pd.DataFrame(data)

    return df


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

    def save_checkpoint(self, agent, directory, current_task):
        path = directory+f"{self.exp_id}-t{current_task}-s{self.timer}.pth"
        logging.info(f"Saving model checkpoint to {path}")
        info = {'step': self.timer}
        
        save_model_checkpoint(agent.network, agent.optimizer, path, **info, config=agent.config)

    @staticmethod
    def get_latest_checkpoint_path(directory, exp_id, task=None):
            checkpoint_paths = glob.glob(directory + f"{exp_id}-t{task if task is not None else '*'}-s*.pth")
            if not checkpoint_paths:
                return None
            latest_checkpoint = max(checkpoint_paths, key=os.path.getctime)
            return latest_checkpoint
    
    def load_checkpoint(self, agent, directory, task_to_load):
        path = self.get_latest_checkpoint_path(directory, self.exp_id, task=task_to_load)
        if path:
            logging.info(f"Loading model checkpoint from {path}")
            info, config = load_model_checkpoint(agent.model, agent.optimizer, path, ["step"])
            self.timer = info['step']
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
            # create summary statistics 
            # renaming the keys before logging
            final_res = {}
            for k,v in end_results.items():
                final_res[f'{k}_end'] = v
            logging.info(f"Final evaluation {self.agent_type} Agent: {final_res}")

            # Log to external JSON file
            self.current_experiment['end_results'] = final_res


        # Close WandB
        if wandb.run is not None:
            # we need to add the agent_name suffix
            wandb.log(final_res)
    


class ExperimentLogger:

    def __init__(self, exp_id, exp_timestamp, exp_name, log_to_file=False, external_json_file=None, log_wandb=False, wandb_project = None, config=None):
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

    
    def save_checkpoint(self, agent, current_task):
        directory = f'{self.exp_directory}/checkpoints/'
        os.makedirs(directory, exist_ok=True)
        self.agent_logger.save_checkpoint(agent, directory, current_task)

    def load_checkpoint(self, agent, task_to_load):
        directory = f'{self.exp_directory}/checkpoints/'
        agent = self.agent_logger.load_checkpoint(agent, directory, task_to_load)
        return agent

    

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

