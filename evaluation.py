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

import torch
from torch.cuda.amp import autocast

def hessian_vector_product(grad, params, vectors):
    """Compute batched Hessian-vector product (HVP) using autograd."""
    hvps = torch.autograd.grad(grad, params, grad_outputs=vectors, retain_graph=True)
    return torch.stack([g.flatten() for g in hvps], dim=1)  # Stack for batched computation

def top_k_hessian_eigen(agent, criterion, data, K=5, max_iters=20,ntasks_observed=-1):
    """GPU-optimized computation of top-K Hessian eigenvalues and eigenvectors using Lanczos."""
    agent.network.zero_grad()
    x, y, task_id = data
    x, y = x.cuda(), y.cuda()
    
    
    with autocast():  # Mixed precision for better performance
        out,y = agent.predict((x,y,task_id), ntasks_observed)
        loss = criterion(out, y)
    
    grads = torch.autograd.grad(loss, agent.network.parameters(), create_graph=True)
    grad_vector = torch.cat([g.flatten() for g in grads]).detach()  # Convert to vector
    N = grad_vector.shape[0]  # Number of parameters
    
    # Initialize Krylov subspace
    Q = torch.zeros((N, K), device='cuda', dtype=torch.float32)
    T = torch.zeros((K, K), device='cuda', dtype=torch.float32)
    
    q = torch.randn(N, device='cuda', dtype=torch.float32)
    q /= q.norm()
    
    for k in range(K):
        Q[:, k] = q
        Hv = hessian_vector_product(grad_vector, list(agent.network.parameters()), q)
        
        if k > 0:
            Hv -= T[k - 1, k] * Q[:, k - 1]  # Orthogonalization
        
        alpha = torch.dot(Hv, q)
        T[k, k] = alpha
        Hv -= alpha * q
        
        beta = Hv.norm()
        if k < K - 1:
            T[k, k + 1] = beta
            T[k + 1, k] = beta
            q = Hv / beta
    
    # Compute eigenvalues and eigenvectors of the small KxK matrix
    eigvals, eigvecs = torch.linalg.eigh(T.cpu())
    eigvecs_full = Q @ eigvecs.cuda()  # Map back to full space
    
    return eigvals, eigvecs_full
