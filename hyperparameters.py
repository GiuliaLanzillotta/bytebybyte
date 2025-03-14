"""
We keep all the experiment hyperparameters in a dictionary with an entry for each dataset and network used. 
If there is no entry corresponding to a specific dataset and network the default values will be used instead.
"""
agent_hyperparameters = {
    "split-c100-chunks-10": {
        "lr":0.05,
        "batch_size":50, #1 epoch = 100 steps
        "optimizer":"SGD",
        "weight_decay":1e-5,
        "momentum":0.9,
        "scheduler_type":"cosine_anneal",
        "warmup_on":True
    }
}