# HW1


You might need to set up pixi, and do 
``` bash
pixi shell
```


## (Q1.3) — behavior cloning, Ant-v2 and Walker2d-v2

```bash
python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --eval_batch_size 5000 --video_log_freq -1 --no_gpu

python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Walker2d.pkl \
    --env_name Walker2d-v2 --exp_name bc_walker --n_iter 1 \
    --expert_data rob831/expert_data/expert_data_Walker2d-v2.pkl \
    --eval_batch_size 5000 --video_log_freq -1 --no_gpu

```

## (Q1.4) — BC return vs gradient steps on Walker2d-v2, 3 seeds


```bash
for SEED in 1 2 3; do
  for S in 100 250 500 1000 2000 5000 10000; do
    python rob831/scripts/run_hw1.py \
        --expert_policy_file rob831/policies/experts/Walker2d.pkl \
        --env_name Walker2d-v2 --exp_name bc_walker_s${S}_seed${SEED} --n_iter 1 \
        --expert_data rob831/expert_data/expert_data_Walker2d-v2.pkl \
        --num_agent_train_steps_per_iter $S --seed $SEED \
        --eval_batch_size 5000 --video_log_freq -1 --no_gpu
  done
done
```


## (Q2.2) — DAgger on Ant-v2 and Walker2d-v2

```bash
python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name dagger_ant --n_iter 10 --do_dagger \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --eval_batch_size 5000 --video_log_freq -1 --no_gpu

python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Walker2d.pkl \
    --env_name Walker2d-v2 --exp_name dagger_walker --n_iter 10 --do_dagger \
    --expert_data rob831/expert_data/expert_data_Walker2d-v2.pkl \
    --eval_batch_size 5000 --video_log_freq -1 --no_gpu

```
