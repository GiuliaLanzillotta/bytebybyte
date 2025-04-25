source clenv/bin/activate


# python experiment.py --device $1 --steps_per_task 1000 --number_tasks 10 --environment split --split_type classes --exp_id exp2 --checkpoint_freq 1000 --network_name resnet18 --agent_type replay --wandb-project order2-approximations --multihead --experiment_type singletask --replay_fraction 0.5 --replay_type "balanced"

# python experiment.py --device $1 --steps_per_task 1000 --number_tasks 10 --environment split --split_type classes --exp_id exp2 --checkpoint_freq 1000 --network_name resnet18 --agent_type replay --wandb-project order2-approximations --multihead --experiment_type singletask --replay_fraction 0.1 --replay_type "balanced"

# python experiment.py --device $1 --steps_per_task 1000 --number_tasks 10 --environment split --split_type classes --exp_id exp2 --checkpoint_freq 1000 --network_name resnet18 --agent_type replay --wandb-project order2-approximations --multihead --experiment_type singletask --replay_fraction 0.01 --replay_type "balanced"

# python experiment.py --device $1 --steps_per_task 1000 --number_tasks 10 --environment split --split_type classes --exp_id exp2-nrt2 --checkpoint_freq 1000 --network_name resnet18 --agent_type replay --wandb-project order2-approximations --multihead --experiment_type singletask --replay_fraction 0.1 --replay_type "fixed" --num_replay_tasks 5

# python experiment.py --device $1 --steps_per_task 1000 --number_tasks 10 --environment split --split_type classes --exp_id exp2-nrt2 --checkpoint_freq 1000 --network_name resnet18 --agent_type replay --wandb-project order2-approximations --multihead --experiment_type singletask --replay_fraction 0.1 --replay_type "fixed" --num_replay_tasks 2


# python experiment.py --device $1 --steps_per_task 1000 --number_tasks 10 --environment split --split_type classes --exp_id exp2-nrt2 --checkpoint_freq 1000 --network_name resnet18 --agent_type replay --wandb-project order2-approximations --multihead --experiment_type singletask --replay_fraction 0.5 --replay_type "fixed" --num_replay_tasks 2


python experiment.py --device $1 --steps_per_task 1000 --number_tasks 10 --environment split --split_type classes --exp_id exp2-nrt2 --checkpoint_freq 1000 --network_name resnet18 --agent_type replay --wandb-project order2-approximations --multihead --experiment_type singletask --replay_fraction 0.999 --replay_type "fixed" --num_replay_tasks 2


python experiment.py --device $1 --steps_per_task 1000 --number_tasks 10 --environment split --split_type classes --exp_id exp2-nrt3 --checkpoint_freq 1000 --network_name resnet18 --agent_type replay --wandb-project order2-approximations --multihead --experiment_type singletask --replay_fraction 0.999 --replay_type "fixed" --num_replay_tasks 3