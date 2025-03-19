source clenv/bin/activate


python experiment.py --device $1 --steps_per_task 1000 --number_tasks 10 --environment split --split_type classes --exp_id exp1sweep --checkpoint_freq 0 --network_name resnet18 --agent_type base --wandb-project order2-approximations --multihead --experiment_type multitask --batch_size 64


python experiment.py --device $1 --steps_per_task 1000 --number_tasks 10 --environment split --split_type classes --exp_id exp1sweep --checkpoint_freq 0 --network_name resnet18 --agent_type base --wandb-project order2-approximations --multihead --experiment_type multitask --batch_size 100


python experiment.py --device $1 --steps_per_task 1000 --number_tasks 10 --environment split --split_type classes --exp_id exp1sweep --checkpoint_freq 0 --network_name resnet18 --agent_type base --wandb-project order2-approximations --multihead --experiment_type multitask --batch_size 150


python experiment.py --device $1 --steps_per_task 1000 --number_tasks 10 --environment split --split_type classes --exp_id exp1sweep --checkpoint_freq 0 --network_name resnet18 --agent_type base --wandb-project order2-approximations --multihead --experiment_type multitask --batch_size 200

# python experiment.py --device $1 --steps_per_task 5000 --number_tasks 10 --environment split --split_type chunks --exp_id exp1-chunks --checkpoint_freq 5000 --network_name resnet18 --agent_type base --wandb-project order2-approximations
