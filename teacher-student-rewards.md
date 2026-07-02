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

```python
#teacher_student_paper2.yml
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

| config | valor | comportamento | métricas |
| --- | --- | --- | --- |
| `sweep_ts_tracklin_0p5.yml` | `0.5` | não anda, colado no chão | [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/TS/runs/qgacg2ai?nw=nwuserluisafrancielle) |
| `sweep_ts_tracklin_1p0.yml` | `1.0` | não anda, cai para frente | [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/TS/runs/v07pjluf?nw=nwuserluisafrancielle) |
| `sweep_ts_tracklin_3p0.yml` | `3.0` | anda, parecido ao baseline | [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/TS/runs/t44xqn2s?nw=nwuserluisafrancielle) |
| `sweep_ts_tracklin_5p0.yml` | `5.0` | anda, mais saltitante / mais esquisito | [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/TS/runs/oianwp30?nw=nwuserluisafrancielle) |

**Conclusão — `track_lin_vel_xy_exp`:** 

## Sweep `track_ang_vel_z_exp`

Recompensa de tracking de **velocidade angular (yaw)**. No baseline o comando de yaw é
**0** (anda só pra frente), então na prática esse termo premia **manter o giro nulo** —
andar reto, sem girar o corpo.

**Registro**:

```python
"track_ang_vel_z_exp": RewTerm(
    func=mdp.track_ang_vel_z_exp,
    weight=weights["track_ang_vel_z_exp"],            # 0.75 no baseline TS
    params={"command_name": "base_velocity", "std": math.sqrt(0.25)},  # std = 0.5
)
```

**Implementação** (Isaac Lab):

```python
def track_ang_vel_z_exp(env, std, command_name, asset_cfg):
    ang_vel_error = torch.square(command[:, 2] - root_ang_vel_b[:, 2])  # (Δω)²
    return torch.exp(-ang_vel_error / std**2)
```

`r = exp( -(Δω)² / std² ) = exp( -4·(Δω)² )`

onde:
* `Δω = ω_comando - ω_real` (yaw), no **body frame**;
* `std = 0.5`, então `std² = 0.25`;
* como o comando de yaw é **0**, o termo recompensa **não girar**;
* peso baseline = `0.75` (metade do `track_lin`).

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

| config | valor | comportamento | métricas |
| --- | --- | --- | --- |
| `sweep_ts_trackang_0p25.yml` | `0.25` | não anda — cai de lado | |
| `sweep_ts_trackang_1p5.yml` | `1.5` | não anda — trava no chão — | |
| `sweep_ts_trackang_3p0.yml` | `3.0` | não anda - trava no chão, uma pata da frente levantada | |

**Conclusão — `track_ang_vel_z_exp`:** muito mais sensível que o `track_lin` — qualquer
desvio do baseline (`0.75`) já quebra a caminhada. Abaixo (`0.25`) o
controle de yaw fica fraco e o robô cai de lado. Acima
(`1.5`, `3.0`) premia fortemente o
"yaw zero" e é melhor ficando imóvel (parado = giro nulo = recompensa maior),
então o robô trava no chão. Só o baseline `0.75` anda. O `track_ang` tem uma janela estreita em torno do baseline.

## Sweep `feet_air_time`

Premia **passos marcados**: recompensa cada pé por ficar mais tempo no ar antes de tocar o
chão. É o termo que combate o "arrastar".

**Registro**:

```python
"feet_air_time": RewTerm(
    func=feet_air_time,
    weight=weights["feet_air_time"],                  # 1.0 no baseline TS
    params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        "command_name": "base_velocity",
        "threshold": 0.5,
    },
)
```

**Implementação** (`go2_mdp.py`):

```python
def feet_air_time(env, sensor_cfg, command_name, threshold):
    # tempo no ar de cada pé até o primeiro contato
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # zera quando o comando é ~0 (robô parado)
    return reward * (norm(command[:, :2]) > 0.1)
```

`r = Σ_pé (t_ar - threshold) · primeiro_contato`

onde:
* premia o pé ficar **mais de `threshold` = 0.5 s** no ar antes de pousar;
* só conta no **primeiro contato** (quando o pé toca);
* **zera com comando ~0** (não premia bater o pé parado).

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

| config | valor | comportamento | métricas |
| --- | --- | --- | --- |
| `sweep_ts_feetair_0p5.yml` | `0.5` | | |
| `sweep_ts_feetair_2p0.yml` | `2.0` | | |

## Sweep `flat_orientation_l2`

**Penalidade** de inclinação do tronco: mede o quanto o corpo está longe de nivelado —
quanto mais inclina, mais perde.

**Registro**:

```python
"flat_orientation_l2": RewTerm(
    func=mdp.flat_orientation_l2,
    weight=weights["flat_orientation_l2"],            # -2.0 no baseline TS
)
```

**Implementação** (Isaac Lab):

```python
def flat_orientation_l2(env, asset_cfg):
    # gravidade projetada no body frame: (g_x, g_y) = 0 quando nivelado
    return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
```

`r = g_x² + g_y²`

onde:
* `(g_x, g_y)` = componentes horizontais da gravidade no **body frame**;
* vale `0` com o tronco perfeitamente nivelado e cresce com a inclinação;
* peso **negativo** (penalidade): forte demais → robô prioriza ficar plano e pode **sentar/congelar**.

### `flat_orientation_l2 = -0.25`

<div class="video-pair">
  <div class="video-panel">
    <p><strong>Teacher</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-teacher-flatorient_0p25.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>Student</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-student-flatorient_0p25.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

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

| config | valor | comportamento | métricas |
| --- | --- | --- | --- |
| `sweep_ts_flatorient_0p25.yml` | `-0.25` | | |
| `sweep_ts_flatorient_1p0.yml` | `-1.0` | | |
| `sweep_ts_flatorient_5p0.yml` | `-5.0` | | |

## Sweep `is_alive`

**Recompensa de sobrevivência**: paga um valor fixo a cada step em que o robô **não
terminou** (não caiu / não resetou). No baseline TS está **desligada (`0.0`)** — de
propósito, porque tende a criar o ótimo local "sobreviver parado".

**Registro**:

```python
"is_alive": RewTerm(
    func=mdp.is_alive,
    weight=weights["is_alive"],                       # 0.0 no baseline TS
)
```

**Implementação** (Isaac Lab):

```python
def is_alive(env):
    return (~env.termination_manager.terminated).float()
```

onde:
* retorna `1.0` em todo step **vivo**, `0.0` no step em que termina;

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

| config | valor | comportamento | métricas |
| --- | --- | --- | --- |
| `sweep_ts_isalive_0p25.yml` | `0.25` | anda — patas mais soltas, cada uma difere da outra | |
| `sweep_ts_isalive_0p5.yml` | `0.5` | anda — galopa em sincronia | |

**Conclusão — `is_alive`:** ao contrário do `track_ang`, não quebra a caminhada nesta faixa —
o robô continua andando nos dois valores. O efeito é na coordenação da passada: em `0.25`
as patas ficam mais soltas e assimétricas (cada uma difere da outra); em `0.5` o "galope" fica
sincronizado. 

## Sweep `action_rate_l2`

Penaliza mudanças bruscas entre ações consecutivas. O peso é negativo: quanto maior a
magnitude, mais a política é pressionada a suavizar os comandos.

### `action_rate_l2 = -0.02`

<div class="video-pair">
  <div class="video-panel">
    <p><strong>Teacher</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-teacher-actionrate_0p002.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>Student</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-student-actionrate_0p002.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

### `action_rate_l2 = -0.05`

<div class="video-pair">
  <div class="video-panel">
    <p><strong>Teacher</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-teacher-actionrate_0p05.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>Student</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-student-actionrate_0p05.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

| config | valor | comportamento | métricas |
| --- | --- | --- | --- |
| `sweep_ts_actionrate_0p02.yml` | `-0.02` | | |
| `sweep_ts_actionrate_0p05.yml` | `-0.05` | | |

## Sweep `dof_torques_l2`

Penaliza esforço de torque nas juntas. O peso é negativo: aumentar a magnitude tende a
favorecer movimentos mais econômicos, mas pode tirar força da locomoção.

### `dof_torques_l2 = -0.0001`

<div class="video-pair">
  <div class="video-panel">
    <p><strong>Teacher</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-260014080-teacher-torques_1em4.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>Student</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-student-torques_1em4.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

### `dof_torques_l2 = -0.0004`

<div class="video-pair">
  <div class="video-panel">
    <p><strong>Teacher</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-teacher-torques_4em4.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>Student</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-student-torques_4em4.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

| config | valor | comportamento | métricas |
| --- | --- | --- | --- |
| `sweep_ts_torques_1em4.yml` | `-0.0001` | | |
| `sweep_ts_torques_4em4.yml` | `-0.0004` | | |

## Sweep `fall_penalty`

Penaliza episódios que terminam em queda/reset. Peso mais negativo aumenta o custo de cair;
com peso zero, a queda deixa de ser punida diretamente por esse termo.

### `fall_penalty = 0.0`

<div class="video-pair">
  <div class="video-panel">
    <p><strong>Teacher</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-teacher-fall_0p0.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>Student</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-student-fall_0p0.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

### `fall_penalty = -2.0`

<div class="video-pair">
  <div class="video-panel">
    <p><strong>Teacher</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-299925504-teacher-fall_2p0.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>Student</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-299925504-student-fall_2p0.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

### `fall_penalty = -4.0`

<div class="video-pair">
  <div class="video-panel">
    <p><strong>Teacher</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-teacher-fall_4p0.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>Student</strong></p>

    <video controls preload="metadata">
      <source src="{{ '/videos/policy-step-280068096-student-fall_4p0.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

| config | valor | comportamento | métricas |
| --- | --- | --- | --- |
| `sweep_ts_fall_0p0.yml` | `0.0` | | |
| `sweep_ts_fall_2p0.yml` | `-2.0` | | |
| `sweep_ts_fall_4p0.yml` | `-4.0` | | |

## Interpretação geral
