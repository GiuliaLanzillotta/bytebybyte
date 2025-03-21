"""
Script with all evaluation functionalities.
"""

import copy
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from scipy.sparse.linalg import LinearOperator, eigsh


from utils import get_cosine_similarity, get_norm_distance, has_batch_norm


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

def get_distance_minima(parameters_list):
    """Accepts a list of network parameters (vectors) and returns the pairwise L2 and cosine distances between them."""

    L = len(parameters_list)

    # Initialize the distance matrices
    l2_distance = np.zeros((L, L))
    cosine_distance = np.zeros((L, L))

    for i in range(L):
        mi = parameters_list[i]
        for j in range(L):
            mj = parameters_list[j]
            l2_distance[i, j] = get_norm_distance(mi, mj)
            cosine_distance[i, j] = get_cosine_similarity(mi, mj)
    
    return l2_distance, cosine_distance

def compute_NTK():
    raise NotImplementedError


def compute_loss(x, y, task_id, criterion, agent):
    out, y_pred = agent.predict((x, y, task_id), -1)
    loss = criterion(out, y_pred)
    return loss

def param_extract_fn(net): 
    return [p for p in net.parameters() if p.requires_grad] 

def hessian_vector_product(vector, x, y, task_id, criterion, agent):
    agent.network.zero_grad()
    active_params = param_extract_fn(agent.network)
    grad_params = torch.autograd.grad(compute_loss(x, y, task_id, criterion, agent), active_params, create_graph=True, allow_unused=True)
    # handle unused parameters 
    grad_params = [g if g is not None else torch.zeros_like(p) for g, p in zip(grad_params, active_params)]
    flat_grad = torch.cat([g.view(-1) for g in grad_params])

    grad_vector_product = torch.sum(flat_grad * vector)
    hvp = torch.autograd.grad(grad_vector_product, active_params, retain_graph=True, allow_unused=True, only_inputs=True)
    hvp = [(g if g is not None else torch.zeros_like(p)) for g, p in zip(hvp, active_params)]
    return torch.cat([g.contiguous().view(-1) for g in hvp])


def matvec(v, x, y, task_id, device, criterion, agent):
    v_tensor = torch.tensor(v, dtype=torch.float32, device=device)
    return hessian_vector_product(v_tensor, x, y, task_id, criterion, agent).cpu().detach().numpy()


def compute_Hessian_spectrum(agent, data, criterion, K=10):  
    agent.ready_train()
    net = agent.network
    device = agent.config['device']
    
    x, y, task_id = data
    x, y = x.to(device), y.to(device)

    num_params = sum(p.numel() for p in param_extract_fn(net))

    linear_operator = LinearOperator((num_params, num_params), matvec=lambda v: matvec(v, x, y, task_id, device, criterion, agent))
    eigenvalues, eigenvectors = eigsh(linear_operator, k=K, tol=0.001, which='LM', return_eigenvectors=True)
    eigenvectors = np.transpose(eigenvectors)

    return eigenvalues[::-1], eigenvectors[::-1] # descending order

def compute_alignment_updates_spectra(task_id, parameters_list, eigenvecs, device):
    L = len(parameters_list)
    K = eigenvecs.shape[0] # eigenvectors on the rows
    all_alignments = np.zeros((L, K))
    random_alignments = np.zeros((L, K))
    start = parameters_list[task_id] # starting value
    for i in range(L):
        end = parameters_list[i] # ending value 
        direction = end - start
        random_vector = torch.randn_like(direction)
        all_alignments[i,:] = [torch.nn.functional.cosine_similarity(direction, torch.Tensor(v).to(device), dim=0).item() for v in eigenvecs]
        # compute random alignments for baseline
        random_alignments[i,:] =  [torch.nn.functional.cosine_similarity(random_vector, torch.Tensor(v).to(device), dim=0).item() for v in eigenvecs]
    return all_alignments, random_alignments





# Inspired by https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb 

def gram_linear(x):
  """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  return x @ x.T

def center_gram(gram, unbiased=False):
  """Center a symmetric Gram matrix.

  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.

  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

  Returns:
    A symmetric matrix with centered columns and rows.
  """
  if not torch.allclose(gram, gram.T):
    raise ValueError('Input must be a symmetric matrix.')
  gram = copy.deepcopy(gram)

  if unbiased:
    # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
    # L. (2014). Partial distance correlation with methods for dissimilarities.
    # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
    # stable than the alternative from Song et al. (2007).
    n = gram.shape[0]
    torch.fill_diagonal(gram, 0)
    means = torch.sum(gram, 0, dtype=torch.float64) / (n - 2)
    means -= torch.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    torch.fill_diagonal(gram, 0)
  else:
    means = torch.mean(gram, 0, dtype=torch.float64)
    means -= torch.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

  return gram


def cka(gram_x, gram_y, debiased=False):
  """Compute CKA.

  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

  Returns:
    The value of CKA between X and Y.
  """
  gram_x = center_gram(gram_x, unbiased=debiased)
  gram_y = center_gram(gram_y, unbiased=debiased)

  # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
  # n*(n-3) (unbiased variant), but this cancels for CKA.
  scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

  normalization_x = torch.linalg.norm(gram_x)
  normalization_y = torch.linalg.norm(gram_y)
  return scaled_hsic / (normalization_x * normalization_y)

def compute_CKA(data, agent, parameters_list,  num_batches=5):
    """ Compute the CKA similarity between the data representations evaluated at different parameter values. 
   - data is a torch dataloader"""
    L = len(parameters_list)
    all_feats = []# L x num_layers x N x D 
    device = agent.config['device']

    for i in range(L): # looping through parameter values 
        agent.network.assign_weights(parameters_list[i])

        # for each parameter value we have a dictionary of {layer -> features}
        features={}; total=0

        for x, _, _ in data: 
            if total >= num_batches: break
            agent.ready_eval()
            x = x.to(device)
            total+=1

            with torch.no_grad():
                feats, _ = agent.network.get_features(x, return_intermediate=True)
                # num_layers x N x D 
            for l, phi in enumerate(feats): 
                if l in features.keys():
                    features[l].append(phi.view(128,-1)) # flatten
                else: features[l] = [phi.view(128,-1)] # flatten
        
        all_feats.append(features)

    # compute the pairwise CKA at every layer of the network  
    # Initialize the distance matrices: one for each layer 
    layers = features.keys()
    cka_distances = [np.zeros((L, L)) for l in range(len(layers))]
    for l in layers:
        for i in range(L): # task i, layer l
            X_i = torch.cat(all_feats[i][l], dim=0)
            for j in range(L): # task j, layer l
                X_j = torch.cat(all_feats[j][l], dim=0)
                cka_distances[l][i,j] = cka(gram_linear(X_i), gram_linear(X_j))
    
    return cka_distances

   
# computing Linear Mode Connectivity

def linear_interpolation(start, end, agent, dataloader, criterion, line_samples=10, tasks_learned=-1, batches=10, return_type="loss"): 
    """ 
    Computes the loss along a linear path between two parameter vectors.
    - return_type (str): loss or error"""
    loss = [] 
    line_range = np.arange(0.0, 1.01, 1.0/float(line_samples))

  
    direction = end - start
    print(direction.max())

    device = agent.config['device']
    
    for t in line_range:
        cur_weight = start + (direction * t)
        agent.network.assign_weights(cur_weight.to(device))
        agent.config['device'] = device
        if has_batch_norm(agent.network): 
            torch.optim.swa_utils.update_bn(iter(dataloader), agent.network, device)
            
        current_loss = evaluate_agent_task(batches, agent, iter(dataloader), criterion, tasks_learned)[f'test_{return_type}']
        loss.append(current_loss)
    return loss, line_range

def compute_LMC_all_toall(parameters_list, agent, dataloader, criterion, line_samples=10, tasks_learned=-1, batches=10, return_type="loss"):
    """ 
    Computes the loss along a linear path between any combination of parameter vectors.
    - return_type (str): loss or error"""
    L = len(parameters_list)
    all_losses = np.zeros((L, L, line_samples+1))
    for i in range(L):
        for j in range(i, L):
            if i == j: continue
            l_values, _ = linear_interpolation(parameters_list[i], parameters_list[j], agent, dataloader, criterion, line_samples, tasks_learned, batches, return_type)
            all_losses[i,j, :] = l_values
    return all_losses

def compute_Fisher():
    raise NotImplementedError
