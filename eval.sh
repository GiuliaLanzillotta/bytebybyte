source clenv/bin/activate

# multitask
python evaluation.py --experiment_name "split-c100-classes-10_resnet18_base_multihead_multitask-exp2-20250320_124048" --device $1  --hessian
#singletask
python evaluation.py --experiment_name "split-c100-classes-10_resnet18_base_multihead_singletask-exp2-20250319_180024" --device $1  --hessian
#replay05 
python evaluation.py --experiment_name "split-c100-classes-10_resnet18_replay_multihead_singletask-exp2-20250319_180051" --device $1  --hessian

#replay01
python evaluation.py --experiment_name "split-c100-classes-10_resnet18_replay_multihead_singletask-exp2-20250319_182618" --device $1  --hessian

#replay001
python evaluation.py --experiment_name "split-c100-classes-10_resnet18_replay_multihead_singletask-exp2-20250319_185604" --device $1  --hessian

#replayfixed
# python evaluation.py --experiment_name "split-c100-classes-10_resnet18_replay_multihead_singletask-exp2-20250319_193440" --device $1  --hessian

#replay nrt2 0.1
# python evaluation.py --experiment_name "split-c100-classes-10_resnet18_replay_multihead_singletask-exp2-nrt2-20250321_180520" --device $1  --hessian


#replay nrt2 0.5
# python evaluation.py --experiment_name "split-c100-classes-10_resnet18_replay_multihead_singletask-exp2-nrt2-20250321_183747" --device $1  --hessian

# replay nrt5 0.1
# python evaluation.py --experiment_name "split-c100-classes-10_resnet18_replay_multihead_singletask-exp2-nrt5-20250321_173230" --device $1  --hessian

#replay nrt1 0.99
python evaluation.py --experiment_name "split-c100-classes-10_resnet18_replay_multihead_singletask-exp2-nrt1-20250325_085156" --device $1  --hessian

#replay nrt2 0.99
python evaluation.py --experiment_name "split-c100-classes-10_resnet18_replay_multihead_singletask-exp2-nrt2-20250410_092627" --device $1  --hessian

#replay nrt3 0.99
python evaluation.py --experiment_name "split-c100-classes-10_resnet18_replay_multihead_singletask-exp2-nrt3-20250410_095646" --device $1  --hessian