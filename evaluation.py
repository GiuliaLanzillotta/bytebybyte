"""
Script with all evaluation functionalities.
"""

import argparse
import copy
import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from scipy.sparse.linalg import LinearOperator, eigsh


from agents import get_agent_class_from_name
from environments import get_environment_from_name
from utils import ExperimentLogger, dotdict, get_cosine_similarity, get_norm_distance, get_params, has_batch_norm, seed_everything
from viz_utils import *



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

def get_Hessian_task(task_id, env, agent, parameters_vec, criterion, N=128, K=10):
    data = env.init_single_task(task_id, train=True)
    dataloader = DataLoader(data, batch_size=N, shuffle=True, num_workers=4)
    data_iterator = iter(dataloader)
    agent.network.assign_weights(parameters_vec)
    if has_batch_norm(agent.network): 
        torch.optim.swa_utils.update_bn(iter(dataloader), agent.network, agent.config['device'])
    eigenvalues, eigenvectors = compute_Hessian_spectrum(agent, next(data_iterator), criterion, K)
    return eigenvalues, eigenvectors

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

def cka_fn(gram_x, gram_y, debiased=False):
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


def get_samples_data(dataloader, num_batches=5): 
    samples = []
    for i, batch in enumerate(dataloader): 
        x, y, t = batch
        samples.append(x)
        if i == num_batches: 
            break
    return torch.cat(samples, dim=0)

def get_features_task(data, agent, parameters_list,  num_batches=5):
    """Computes the layer-wise feature representation of the data (dataloader) for each parameter value in parameters_list.
    - data is a torch dataloader
    - returns a list of dictionaries {layer -> features (NxD)}, one element per parameter value"""
    L = len(parameters_list)
    all_feats = []# L x num_layers x N x D 
    device = agent.config['device']

    # fix a sample of data
    samples = get_samples_data(data, num_batches=num_batches)

    for i in range(L): # looping through parameter values 
        agent.network.assign_weights(parameters_list[i])
        agent.config['device'] = device
        if has_batch_norm(agent.network): 
            torch.optim.swa_utils.update_bn(iter(data), agent.network, device)

        # for each parameter value we have a dictionary of {layer -> features}
        features={}

        for i in range(num_batches): 
            x = samples[i*128:(i+1)*128]
            x = x.to(device)
            agent.ready_eval()

            with torch.no_grad():
                feats, _ = agent.network.get_features(x, return_intermediate=True)
                # num_layers x N x D 
            for l, phi in enumerate(feats): 
                if l in features.keys():
                    features[l].append(phi.view(128,-1)) # flatten
                else: features[l] = [phi.view(128,-1)] # flatten
        
        all_feats.append(features)
    return all_feats

def tasks_CKA_evolution(env, agent, parameters_list,  num_batches=5):
    """ Computes the evolution of the feature representation of every task. It returns a LxL matrix (L= number of tasks), where mat[i,j] is the CKA between the feature representation of task i at parameters 0 and parameters j. 
    - parameters_list[0] is used as the reference parameter value (better to pass initialization)
    - data is a torch dataloader"""
    num_tasks = env.number_tasks
    cka_distances = []
    
    for i in range(num_tasks): 
        data = env.init_single_task(i, train=False)
        dataloader = DataLoader(data, batch_size=128, shuffle=True, num_workers=4)
        task_features = get_features_task(dataloader, agent, parameters_list, num_batches)
        zero_feats = task_features[i+1]
        for j, ij_features in enumerate(task_features[1:]): 
            for l, phi in ij_features.items():
                if len(cka_distances) < l+1: 
                    # initialize the layer CKA matrix
                    cka_distances.append(torch.zeros((num_tasks, num_tasks)))
                X_ij = torch.cat(phi, dim=0)
                X_ii = torch.cat(zero_feats[l], dim=0)
                cka_distances[l][i,j] = cka_fn(gram_linear(X_ij), gram_linear(X_ii))

    
    return cka_distances

   
# computing Linear Mode Connectivity
def linear_interpolation(start, end, agent, dataloader, criterion, line_samples=10, tasks_learned=-1, batches=10, return_type="loss"): 
    """ 
    Computes the loss along a linear path between two parameter vectors.
    - return_type (str): loss or error"""
    loss = [] 
    line_range = np.arange(0.0, 1.01, 1.0/float(line_samples))

  
    direction = end - start

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

def compute_LMC_all2all(tasks_parameters, agent, env, criterion, line_samples=10, tasks_learned=-1, batches=10, return_type="loss"):
    """ 
    Computes the loss along a linear path between any combination of parameter vectors.
    - tasks_parameters: one per task
    - return_type (str): loss or error
    - returns a LxLx(line_samples+1) matrix, where mat[i,j,:] is the loss along the linear path between parameters i and j
    """ 

    num_tasks = env.number_tasks
    assert len(tasks_parameters) == num_tasks, "Number of parameter values must match the number of tasks"
    all_losses = np.zeros((num_tasks, num_tasks, line_samples+1))
    for i in range(num_tasks):
        data = env.init_single_task(i, train=False)
        dataloader = DataLoader(data, batch_size=128, shuffle=True, num_workers=4)
        for j in range(i, num_tasks):
            if i == j: continue
            l_values, _ = linear_interpolation(tasks_parameters[i], tasks_parameters[j], agent, dataloader, criterion, line_samples, tasks_learned, batches, return_type)
            all_losses[i,j,:] = l_values
    return all_losses

def compute_Fisher():
    raise NotImplementedError


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Input parameters from the command line.")
    # COMMAND LINE ARGUMENTS 
    # experiment setup arguments
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of the experiment to evaluate (string)')
    parser.add_argument('--device', type=int, required=True, help='Device identifier (integer).')
    parser.add_argument('--transfer_matrix', action='store_true', help='Make transfer matrix plot.')
    parser.add_argument('--transfer_line', action='store_true', help='Make transfer line plot.')
    parser.add_argument('--distances', action='store_true', help='Compute distances in parameter space (L2, cosine).')
    parser.add_argument('--cka', action='store_true', help='Compute CKA.')
    parser.add_argument('--lmc', action='store_true', help='Compute LMC.')
    parser.add_argument('--hessian', action='store_true', help='Compute Hessian spectrum and alignments.')
    # Parse the arguments
    args = parser.parse_args()


    # Setup 
    exp_name = args.experiment_name
    device = args.device
    device = device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")


    # Load the experiment config 
    directory = f"experiments/{exp_name}/results"
    with open(os.path.join(directory, "experiments.json"), "r") as file:
        data = json.load(file)

    # Extract the "config" dictionary and save it into a pandas dataframe
    config_df = pd.DataFrame([data[0]["config"]])
    acf = dotdict(config_df.iloc[0])
    seed = acf.seed
    acf.device = device
    seed_everything(seed)
    # Add the "end_results" dictionary to the dataframe
    end_results = data[0]["end_results"]
    for key, value in end_results.items():
        config_df[key] = value
    print(acf)

    # Setup the agent, experiment logger, environment
    agent_class, _ = get_agent_class_from_name(acf.agent_type)
    agent = agent_class(**acf)
    agent_config = agent.config
    experiment_logger = ExperimentLogger(acf.exp_id, acf.exp_timestamp, acf.exp_name, config=agent_config)
    env, environment_name = get_environment_from_name(acf.environment, acf)
    batches_eval = env.batches_eval
    eval_criterion = env.criterion
    num_classes_per_task = env.num_classes_per_task 
    num_classes_total = env.num_classes

    if args.transfer_matrix or args.transfer_line: 
        # Load the "transfer" dictionary
        transfer_dict = data[0]["transfer"]
        # Initialize the matrix
        num_tasks = len(transfer_dict)
        
        matrix = np.zeros((num_tasks, num_tasks))
        # Populate the matrix
        for i in range(num_tasks):
            for j in range(num_tasks):
                # removing from 1 to get accuracy rather than error
                matrix[i, j] = 1- transfer_dict[str(i)][str(j)]['test_error']
        
        if args.transfer_matrix: 
            viz_heatmap(matrix, "Transfer", "Target Task", "Source Task", os.path.join(directory, "transfer_matrix.pdf"))
        
        if args.transfer_line: 
            viz_lineplots(matrix, "Transfer", "Target Task", "Test Accuracy", os.path.join(directory, "transfer_line.pdf"), lbl_names="Task")

    all_minima = []
    # adding initialization if present
    agent = experiment_logger.load_checkpoint(agent, "init", 0)
    all_minima.append(get_params(agent.network).to(device))
    for i in range(num_tasks):
        agent = experiment_logger.load_checkpoint(agent, i)
        all_minima.append(get_params(agent.network).to(device))
    agent.config['device']=device

    if args.distances:
        l2_mat, cosine_mat = get_distance_minima(all_minima)
        experiment_logger.log_matrix(l2_mat, "l2_distance", plot=True, title="L2 distances", xaxis="Task (checkpoint)", yaxis="Task (checkpoint)")
        experiment_logger.log_matrix(cosine_mat, "cosine_sim", plot=True, title="Cosine similarity", xaxis="Task (checkpoint)", yaxis="Task (checkpoint)")
        
    if args.cka:
        cka_distance_mats = tasks_CKA_evolution(env, agent, all_minima, num_batches=batches_eval) # a list of layer-wise feature evolution matrices (task data x task number)
        for l, mat in enumerate(cka_distance_mats):
            # saving and plotting CKA distances 
            experiment_logger.log_matrix(mat, f"cka_distance_layer_{l}")
            experiment_logger.plot_lines(mat, title=f"CKA evolution Layer {l}", xaxis="Task (checkpoint)", yaxis=f"CKA similarity to learned features", name=f"cka_distance_layer_{l}", lbl_names=f"Task")

    if args.lmc:
        LMC_matrices = compute_LMC_all2all(all_minima[1:], agent, env, eval_criterion, line_samples=10, batches=batches_eval, return_type="loss") # (num_tasks, num_tasks, line_samples+1) matrix
        experiment_logger.log_matrix(LMC_matrices, f"LMC_all_tasks")
        # plot the LMC from the first task to all other tasks
        experiment_logger.plot_lines(LMC_matrices[0,:,:], title=f"LMC Task 0", xaxis="Interpolation", yaxis=f"Validation Loss Task {l}", name=f"LMC_task_0", lbl_names=f"{l+1} -", ylim=(0.0,5.0))

    if args.hessian:    # Hessian computation
        for i in range(num_tasks):
            eigval, eigvec = get_Hessian_task(i, env, agent, all_minima[i+1], eval_criterion, N=128, K=10)
            experiment_logger.log_matrix(eigval, f"eigval_task_{i}")
            experiment_logger.log_matrix(eigvec, f"eigvec_task_{i}")
            experiment_logger.plot_lines(eigval.reshape(-1, 1), title=f"Spectrum Task {i}", xaxis="Index", yaxis=f"Eigenvalue", name=f"Spectrum_task{l}", lbl_names="no")
            # plotting eigval
            if i==0: 
                all_alignments, random_alignments = compute_alignment_updates_spectra(0, all_minima[1:], eigvec, device)
                # plotting alignments 
                experiment_logger.plot_lines(all_alignments, title="Cosine Similarity of Task Displacement with Eigenvectors", xaxis="Eigenvector index", yaxis="Alignment", name="alignments", lbl_names="Displacement Task")


    print("Evaluation completed.")