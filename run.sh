source clenv/bin/activate

python experiment.py --device 0 --steps_per_task 1000 --number_tasks 10 --environment split --split_type classes --exp_id exp1 --checkpoint_freq 1000 --network_name resnet18 --agent_type base --wandb-project order2-approximations --multihead


python experiment.py --device 0 --steps_per_task 1000 --number_tasks 10 --environment split --split_type classes --exp_id exp1 --checkpoint_freq 1000 --network_name resnet18 --agent_type base --wandb-project order2-approximations --singlehead


python experiment.py --device 0 --steps_per_task 3000 --number_tasks 10 --environment split --split_type classes --exp_id exp1-longertask --checkpoint_freq 3000 --network_name resnet18 --agent_type base --wandb-project order2-approximations --multihead


python experiment.py --device 0 --steps_per_task 1000 --number_tasks 20 --environment split --split_type classes --exp_id exp1-moretasks --checkpoint_freq 1000 --network_name resnet18 --agent_type base --wandb-project order2-approximations --multihead


python experiment.py --device 0 --steps_per_task 1000 --number_tasks 10 --environment split --split_type chunks --exp_id exp1-chunks --checkpoint_freq 1000 --network_name resnet18 --agent_type base --wandb-project order2-approximations