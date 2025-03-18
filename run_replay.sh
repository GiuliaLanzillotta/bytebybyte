source clenv/bin/activate


python experiment.py --device 1 --steps_per_task 1000 --number_tasks 10 --environment split --split_type classes --exp_id exp1 --checkpoint_freq 0 --network_name resnet18 --agent_type replay --wandb-project order2-approximations --multihead --experiment_type singletask --replay_fraction 0.1 --replay_type "balanced"
