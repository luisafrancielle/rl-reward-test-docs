---
layout: default
title: Rewards · Teacher–Student
section: teacher-student-rewards
kicker: TEACHER–STUDENT
description: Documentação da função de recompensa utilizada no cenário Teacher–Student.
permalink: /teacher-student-rewards/
---

# Anotações Teacher–Student

Config base para avaliação das mudanças de parâmetros em Teacher–Student. Ele não anda muito bem, mas foi o primeiro que consegui fazer andar.

## Hiperparâmetros do PPO

| parâmetro | valor |
| --- | --- |
| `learning_rate` | `1.0e-3` |
| `num_envs` | `4096` |
| `num_steps` | `24` |
| `anneal_lr` | `false` |
| `gamma` | `0.99` |
| `gae_lambda` | `0.95` |
| `num_minibatches` | `4` |
| `update_epochs` | `5` |
| `norm_adv` | `true` |
| `clip_coef` | `0.2` |
| `clip_vloss` | `true` |
| `ent_coef` | `0.01` |
| `vf_coef` | `1.0` |
| `max_grad_norm` | `1.0` |
| `adaptive_lr` | `true` |
| `desired_kl` | `0.01` |
| `lr_min` | `1.0e-5` |
| `lr_max` | `1.0e-2` |
| `target_kl` | `0.01` |

```python
#teacher_student_papder2.yml
use_teacher_student: true

seed: 42
torch_deterministic: true
cuda: true

track: true
video: true
video_length: 800
video_resolution: "1280x720"
video_num_envs: 18
video_poll_seconds: 300

total_timesteps: 300000000
checkpoint_every: 20000000
checkpoint_max_to_keep: 5
save_optimizer_in_ckpt: false

# PPO — paper Tab. V 
learning_rate: 1.0e-3
num_envs: 4096
num_steps: 24
anneal_lr: false
gamma: 0.99
gae_lambda: 0.95
num_minibatches: 4
update_epochs: 5
norm_adv: true
clip_coef: 0.2
clip_vloss: true
ent_coef: 0.01
vf_coef: 1.0
max_grad_norm: 1.0

adaptive_lr: true
desired_kl: 0.01
lr_min: 1.0e-5
lr_max: 1.0e-2
target_kl: 0.01

use_obs_norm: true
obs_norm_epsilon: 1e-8

obs_history_length: 15
latent_dim: 8
priv_dim: 18
adaptation_lr: 1.0e-3

env_kwargs:
  scene: flat
  spacing: 3.0
  safety_margin: 0.1

  rewards:
    track_lin_vel_xy_exp: 1.5
    track_ang_vel_z_exp: 0.75
    feet_air_time: 1.0

    flat_orientation_l2: -2.0
    dof_pos_limits: -2.0

    lin_vel_z_l2: -2.0
    ang_vel_xy_l2: -0.05
    dof_torques_l2: -0.0002
    dof_acc_l2: -2.5e-7
    action_rate_l2: -0.01

    is_alive: 0.0
    fall_penalty: -1.0
    illegal_contact_penalty: -1.0
    head_contact_penalty: -1.0
    undesired_contacts: 0.0
    feet_air_time_rear: 0.0
```

## Vídeo baseline

<div class="video-pair">
  <div class="video-panel">
    <p><strong>Teacher</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-190021632-teacher.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>Student</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-190021632-student.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

## Recompensa avaliada

Nesta rodada, o sweep foi feito apenas em `track_lin_vel_xy_exp`. O objetivo é entender como o peso dessa recompensa muda a marcha do **teacher** e o quanto o **student** consegue preservar esse comportamento.

**Registro**:

```python
"track_lin_vel_xy_exp": RewTerm(
    func=mdp.track_lin_vel_xy_exp,
    weight=weights["track_lin_vel_xy_exp"],            # 1.5 no baseline TS
    params={"command_name": "base_velocity", "std": math.sqrt(0.25)},  # std = 0.5
)
```

**Implementação** (Isaac Lab):

```python
def track_lin_vel_xy_exp(env, std, command_name, asset_cfg):
    lin_vel_error = torch.sum(
        torch.square(command[:, :2] - root_lin_vel_b[:, :2]),  # (Δvx)² + (Δvy)²
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)
```

`r = exp( -[ (Δvx)² + (Δvy)² ] / std² )`

onde:
* `Δv = v_comando - v_real`, medido no **body frame**;
* `std = 0.5`, então `std² = 0.25`;
* a função fica perto de `1.0` quando o robô segue bem o comando;
* o peso no config define quanto esse acerto de velocidade entra na soma total da recompensa.


## Sweep `track_lin_vel_xy_exp`

### `track_lin_vel_xy_exp = 0.5`

<div class="video-pair">
  <div class="video-panel">
    <p><strong>Teacher</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-299925504-teacher-tracklin_0p5.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>Student</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-299925504-student-tracklin_0p5.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

### `track_lin_vel_xy_exp = 1.0`

<div class="video-pair">
  <div class="video-panel">
    <p><strong>Teacher</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-teacher-tracklin-1p0.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>Student</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-student-tracklin-1p0.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

### `track_lin_vel_xy_exp = 3.0`


<div class="video-pair">
  <div class="video-panel">
    <p><strong>Teacher</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-teacher-tracklin-3p0.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>Student</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-student-tracklin-3p0.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

### `track_lin_vel_xy_exp = 5.0`

<div class="video-pair">
  <div class="video-panel">
    <p><strong>Teacher</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-teacher-tracklin-5p0.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>Student</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-student-tracklin-5p0.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

| config | valor | Métricas |
| --- | --- | --- |
| `sweep_ts_tracklin_0p5.yml` | `0.5` | |
| `sweep_ts_tracklin_1p0.yml` | `1.0` | |
| `sweep_ts_tracklin_3p0.yml` | `3.0` | |
| `sweep_ts_tracklin_5p0.yml` | `5.0` | |

## Sweep `track_ang_vel_z_exp`

### `track_ang_vel_z_exp = 0.25`

<div class="video-pair">
  <div class="video-panel">
    <p><strong>Teacher</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-teacher-trackang_0p25.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>Student</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-student-trackang_0p25.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

### `track_ang_vel_z_exp = 1.5`

<div class="video-pair">
  <div class="video-panel">
    <p><strong>Teacher</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-260014080-teacher-trackang_1p5.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>Student</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-student-trackang_1p5.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

### `track_ang_vel_z_exp = 3.0`

<div class="video-pair">
  <div class="video-panel">
    <p><strong>Teacher</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-teacher-trackang_3p0.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>Student</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-student-trackang_3p0.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

| config | valor | Métricas |
| --- | --- | --- |
| `sweep_ts_trackang_0p25.yml` | `0.25` | |
| `sweep_ts_trackang_1p5.yml` | `1.5` | |
| `sweep_ts_trackang_3p0.yml` | `3.0` | |

## Sweep `feet_air_time`

### `feet_air_time = 0.5`

<div class="video-pair">
  <div class="video-panel">
    <p><strong>Teacher</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-299925504-teacher-feetair_0p5.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>Student</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-299925504-student-feetair_0p5.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

### `feet_air_time = 2.0`

<div class="video-pair">
  <div class="video-panel">
    <p><strong>Teacher</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-teacher-feetair_2p0.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>Student</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-student-feetair_2p0.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

| config | valor | Métricas |
| --- | --- | --- |
| `sweep_ts_feetair_0p5.yml` | `0.5` | |
| `sweep_ts_feetair_2p0.yml` | `2.0` | |

## Sweep `flat_orientation_l2`

### `flat_orientation_l2 = -1.0`

<div class="video-pair">
  <div class="video-panel">
    <p><strong>Teacher</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-teacher-flatorient_1p0.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>Student</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-student-flatorient_1p0.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

### `flat_orientation_l2 = -5.0`

<div class="video-pair">
  <div class="video-panel">
    <p><strong>Teacher</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-299925504-teacher-flatorient_5p0.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>Student</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-299925504-student-flatorient_5p0.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

| config | valor | Métricas |
| --- | --- | --- |
| `sweep_ts_flatorient_1p0.yml` | `-1.0` | |
| `sweep_ts_flatorient_5p0.yml` | `-5.0` | |

## Sweep `is_alive`

### `is_alive = 0.25`

<div class="video-pair">
  <div class="video-panel">
    <p><strong>Teacher</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-teacher-isalive_0p25.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>Student</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-student-isalive_0p25.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

### `is_alive = 0.5`

<div class="video-pair">
  <div class="video-panel">
    <p><strong>Teacher</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-teacher-isalive_0p5.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>Student</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-student-isalive_0p5.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

| config | valor | Métricas |
| --- | --- | --- |
| `sweep_ts_isalive_0p25.yml` | `0.25` | |
| `sweep_ts_isalive_0p5.yml` | `0.5` | |

## Interpretação geral
