source clenv/bin/activate

python experiment.py --device 0 --seed 11 --steps_per_task 1000 --number_tasks 10 --environment split --split_type classes --exp_id try --checkpoint_freq 1000 --permutation_size 32 --network_name resnet18 --agent_type base --wandb-project order2-approximations --multihead