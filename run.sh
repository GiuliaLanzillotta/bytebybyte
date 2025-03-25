source clenv/bin/activate


# python experiment.py --device $1 --steps_per_task 1000 --number_tasks 10 --environment split --split_type classes --exp_id exp2 --checkpoint_freq 1000 --network_name resnet18 --agent_type base --wandb-project order2-approximations --multihead --experiment_type singletask

python experiment.py --device 1 --steps_per_task 100 --number_tasks 10 --environment split --split_type classes --exp_id exp2 --checkpoint_freq 0 --network_name resnet18 --agent_type base --wandb-project order2-approximations --multihead --experiment_type multitask

python experiment.py --device $1 --steps_per_task 10000 --number_tasks 10 --environment split --split_type classes --exp_id exp2-longertask --checkpoint_freq 10000 --network_name resnet18 --agent_type base --wandb-project order2-approximations --multihead --experiment_type multitask

python experiment.py --device $1 --steps_per_task 1000 --number_tasks 20 --environment split --split_type classes --exp_id exp2-moretasks --checkpoint_freq 1000 --network_name resnet18 --agent_type base --wandb-project order2-approximations --multihead --experiment_type multitask


# python experiment.py --device $1 --steps_per_task 5000 --number_tasks 10 --environment split --split_type chunks --exp_id exp1-chunks --checkpoint_freq 5000 --network_name resnet18 --agent_type base --wandb-project order2-approximations
