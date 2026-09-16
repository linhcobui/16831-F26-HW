# HW1 — commands to reproduce every reported number

Environment: Python 3.10, torch 1.12.1, gym 0.25.1, mujoco-py 2.1.2.14 (MuJoCo 2.1 at
`~/.mujoco/mujoco210`). Managed with [pixi](https://pixi.sh): `pixi install`, then
`pixi shell` (equivalent to `conda activate rob831`). All commands are run from `hw1/`.

`--no_gpu` is passed everywhere because the runs were done on an Apple Silicon Mac (no CUDA).
`--video_log_freq -1` disables video logging for every submitted run.

Shared hyperparameters (defaults of `run_hw1.py`, unless a command overrides them):
`--n_layers 2 --size 64 --learning_rate 5e-3 --train_batch_size 100
--num_agent_train_steps_per_iter 1000 --batch_size 1000 --ep_len 1000 --seed 1`,
tanh hidden activations / identity output, Adam, MSE loss.

## Table 1 (Q1.2) — expert return, mean and std over the 2 provided trajectories

No training; reads the demonstration pickles directly.

```bash
python -c "
import pickle, numpy as np
for e in ['Ant','Humanoid','Walker2d','Hopper','HalfCheetah']:
    r = [p['reward'].sum() for p in pickle.load(open(f'rob831/expert_data/expert_data_{e}-v2.pkl','rb'))]
    print(f'{e:12s} mean {np.mean(r):8.1f}  std {np.std(r):6.1f}')"
```

## Table 2 (Q1.3) — behavior cloning, Ant-v2 (>=30% of expert) and Walker2d-v2 (<30%)

```bash
python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --eval_batch_size 5000 --video_log_freq -1 --no_gpu
# -> run_logs/q1_bc_ant_Ant-v2_16-09-2026_11-47-27 : 4697.7 +/- 95.2

python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Walker2d.pkl \
    --env_name Walker2d-v2 --exp_name bc_walker --n_iter 1 \
    --expert_data rob831/expert_data/expert_data_Walker2d-v2.pkl \
    --eval_batch_size 5000 --video_log_freq -1 --no_gpu
# -> run_logs/q1_bc_walker_Walker2d-v2_15-09-2026_12-45-25 : 348.7 +/- 361.9
```

## Figure 1 (Q1.4) — BC return vs gradient steps on Walker2d-v2, 3 seeds

21 runs: 7 values of `num_agent_train_steps_per_iter` x 3 seeds.

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

python scripts/plot_q1_4.py   # -> figures/q1_4_walker_sweep.pdf
```

Seed-mean return: 290 (100 steps), 450 (250), 300 (500), 495 (1000), 2466 (2000),
3140 (5000), 2986 (10000).

## Figure 2 (Q2.2) — DAgger on Ant-v2 and Walker2d-v2

```bash
python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name dagger_ant --n_iter 10 --do_dagger \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --eval_batch_size 5000 --video_log_freq -1 --no_gpu
# -> run_logs/q2_dagger_ant_Ant-v2_16-09-2026_11-47-38 : 4697.7 -> 4790.7

python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Walker2d.pkl \
    --env_name Walker2d-v2 --exp_name dagger_walker --n_iter 10 --do_dagger \
    --expert_data rob831/expert_data/expert_data_Walker2d-v2.pkl \
    --eval_batch_size 5000 --video_log_freq -1 --no_gpu
# -> run_logs/q2_dagger_walker_Walker2d-v2_15-09-2026_12-45-26 : 348.7 -> 5392.5

python scripts/plot_q2.py     # -> figures/q2_dagger.pdf
```

## Reading logged numbers out of the tfevents files

```bash
python -c "
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for d in sorted(glob.glob('data/q*')):
    ea = EventAccumulator(d); ea.Reload()
    if 'Eval_AverageReturn' not in ea.Tags()['scalars']: continue
    g = lambda k: [round(e.value, 1) for e in ea.Scalars(k)]
    print(d.split('/')[-1], '| mean', g('Eval_AverageReturn'), '| std', g('Eval_StdReturn'))"
```

Or interactively: `python -m tensorboard.main --logdir data`.

## Note on Humanoid-v2 (not used in the report)

Humanoid was tried first as the "<30% of expert" task, but its provided expert policy does not
work under MuJoCo 2.x: rolled out live it scores 76.0 with 16-step episodes (the `cfrc_ext`
contact-force observation dimensions read as zero, whereas the expert was trained when they were
populated). DAgger therefore converges to a policy that falls immediately, so Walker2d-v2 was used
instead. Diagnostic:

```bash
python -c "
import gym
from rob831.infrastructure import pytorch_util as ptu; ptu.init_gpu(False)
from rob831.policies.loaded_gaussian_policy import LoadedGaussianPolicy
from rob831.infrastructure.utils import sample_trajectories
import numpy as np
for e in ['Ant','Humanoid','Walker2d','Hopper','HalfCheetah']:
    env = gym.make(e+'-v2'); env.reset(seed=1)
    paths, _ = sample_trajectories(env, LoadedGaussianPolicy(f'rob831/policies/experts/{e}.pkl'), 3000, 1000)
    print(e, np.mean([p['reward'].sum() for p in paths]), np.mean([len(p['reward']) for p in paths]))"
```
