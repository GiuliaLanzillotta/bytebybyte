"""
Script with all evaluation functionalities.
"""


import torch
from torch.utils.data.dataloader import DataLoader


def evaluate_agent_all_tasks_env(num_batches, agent, env, train=False, eval_criterion=None, ntasks_observed=-1):
    """Evaluates the agent on all tasks in the environment."""

    res_offline = {}
    agent.ready_eval()

    if eval_criterion is None: eval_criterion=env.criterion
    if ntasks_observed < 0: ntasks_observed = env.number_tasks
    # Evaluate on all the tasks 
    for i in range(env.number_tasks):
        test_data = env.init_single_task(task_number=i, train=train) 
        test_data_iterator = iter(DataLoader(test_data, batch_size=128, shuffle=True, num_workers=8)) 
        res = evaluate_agent_task(num_batches, agent, test_data_iterator, eval_criterion, ntasks_observed)
        res["task_name"] = env.task_names[i]
        
        res_offline[i] = res
    
    return res_offline

def evaluate_agent_task(num_batches, agent, test_iter, eval_criterion, ntasks_observed):
    # average loss and accuracy on 'num_batches' batches
    
    agent.ready_eval()

    loss = 0; acc=0; total=0
    for b in range(num_batches):
        x, y, task_id = next(test_iter)
        total+=len(y)
        with torch.no_grad():
            out,y = agent.predict((x,y,task_id), ntasks_observed)
            loss += eval_criterion(out, y)
            acc += torch.sum((torch.max(out,dim=1)[1] == y).float())


    loss /=(b+1)
    acc /= total

    results = {
            "test_loss":loss.item(),
            "test_error":(1 - acc.item())
        }
    
    return results


def compute_NTK():
    raise NotImplementedError

def compute_Hessian():
    raise NotImplementedError

def compute_CKA():
    raise NotImplementedError

def compute_Fisher():
    raise NotImplementedError

