---
layout: default
title: Rewards · Base
section: base-rewards
kicker: BASE ARCHITECTURE
description: Documentação da função de recompensa utilizada na arquitetura base.
permalink: /base-rewards/
---

# Anotações Treinos

Config Mestre para avaliação das mudanças de parâmetros.

```python
# Go2 rough-terrain velocity training v18
# Key changes from v16:
#   - Added feet_slide penalty to directly punish foot dragging
#   - Wider command range restored ([-1.5, 1.5]) — dragging fix is structural now
#   - DDP-optimized hyperparameters for 4 GPUs

seed: 42
torch_deterministic: true
cuda: true

track: true

video: true
video_length: 1600
video_mode: "during"
video_resolution: "1280x720"
video_num_envs: 18
video_poll_seconds: 15

total_timesteps: 500000000

checkpoint_every: 20000000
checkpoint_max_to_keep: 5
save_optimizer_in_ckpt: true

learning_rate: 0.001
num_envs: 16384
num_steps: 100
anneal_lr: false
gamma: 0.99
gae_lambda: 0.95
num_minibatches: 16
update_epochs: 5
norm_adv: true
clip_coef: 0.2
clip_vloss: true
ent_coef: 0.008
vf_coef: 0.5
max_grad_norm: 1.0
target_kl: null

use_obs_norm: true
obs_norm_epsilon: 1e-8

curriculum_cfg:
  schedule: cosine # linear cosine exponential step3 step5
  profile: full # forward_only planar full
  warmup_fraction: 0.1

env_kwargs:
  spacing: 3.0
  safety_margin: 0.1
  use_relative_control: False
  relative_scale: 0.1
  action_interval: 0.02
  scene: flat

  # command_lin_vel_x: [-1.5, 1.5]
  # command_lin_vel_y defaults to (0,0) — forward only
  # command_ang_vel_z: [0.0, 0.0]
  static_friction_range: (0.8, 0.8)
  dynamic_friction_range: (0.6, 0.6)

  rewards:
    track_lin_vel_xy_exp: 1.5
    track_ang_vel_z_exp: 0.75

    lin_vel_z_l2: -2.0
    ang_vel_xy_l2: -0.05
    dof_torques_l2: -0.0002
    dof_acc_l2: -2.5e-7
    action_rate_l2: -0.01

    # Contact / gait terms
    feet_air_time: 0.5            # reward stepping (threshold=0.2s in code)
    feet_air_time_rear: 0.0
    feet_slide: -0.25             # NEW — directly penalizes foot velocity while in contact;
                                  # this is the actual fix for dragging: makes sliding feet
                                  # costly regardless of command magnitude
    undesired_contacts: -0.2

    illegal_contact_penalty: -1.0
    fall_penalty: -2.0
    head_contact_penalty: -1.0
    is_alive: 0.5

    flat_orientation_l2: -0.25
    dof_pos_limits: 0.0
```

**Vídeo Baseline**
<div class="video-panel">
  <p><strong>baseline</strong></p>
  <video controls preload="metadata">
    <source src="{{ '/videos/config_policy-step-999424000.mp4' | relative_url }}" type="video/mp4">
  </video>
</div>
Azul : Hipótese feita, os resultados confirmarão, ou não.
## Recompensas
o Isaac Lab multiplica **todo** termo por `dt` (≈0.02). Então a contribuição real por step é `weight × 1.0 × dt`.

```cs
value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
```

O que vai pro wandb (`eval/reward_terms`) é o `value / dt` — ou seja, **sem o dt**, de volta pra `weight × func` = 0.5. Então **o gráfico mostra 0.5, mas a recompensa real é 0.01**. Os gráficos são normalizados "por segundo"; o reward de verdade é por-step (×dt). É por isso que os números do painel parecem maiores do que o que a política realmente otimiza.
Soma das recompensas é feita pelo IsaacLab:

```python
def compute(self, dt):
    self._reward_buf[:] = 0.0
    for name, term_cfg in zip(self._term_names, self._term_cfgs):
        if term_cfg.weight == 0.0:
            continue
        value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
        self._reward_buf += value          # <<< soma de todos os termos
    return self._reward_buf
```

`R_total = Σ_i func_i(env) × weight_i × dt`
onde:
*   a soma é sobre **todos** os termos de recompensa (os que estamos documentando)
*   `weight_i` = peso de cada termo (do config/YAML)
*   `dt ≈ 0.02` (multiplica **todo** termo igualmente)
*   termos com `weight == 0` são pulados (não entram na soma)
É **esse** `R_total` (por env, por step) que vira a recompensa do PPO. Cada `func_i` é uma das recompensas individuais (track\_lin, track\_ang, feet\_slide, is\_alive, etc.).
### is\_alive

```python
def is_alive(env):
    return (~env.termination_manager.terminated).float()
```

Retorna **1.0** em todo step que o robô **não** terminou (não caiu/não resetou), e **0.0** no step em que termina.
Treinos:
Sweep isalive\_0: [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/1uaj7uq0?nw=nwuserimdudak) | Sweep isalive\_1: [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/sivema1e?nw=nwuserimdudak)

<div class="video-pair">
  <div class="video-panel">
    <p><strong>isalive_0</strong></p>
    <video controls preload="metadata">
      <source src="{{ '/videos/sweep_isalive_0_policy-step-999424000.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>isalive_1p0</strong></p>
    <video controls preload="metadata">
      <source src="{{ '/videos/sweep_isalive_1p0_policy-step-999424000.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

Com a recompensa desativada, o robô teve maior dificuldade para aprender a andar. Por outro lado, aumentar a recompensa para 1.0 piorou quase todos os gráficos(ou ficou muito parecido). ![](https://t90132646619.p.clickup-attachments.com/t90132646619/4ac6d146-3b1f-4503-86f1-3db038ac5d0f/image.png)
### track\_lin\_vel\_xy\_exp
**Registro** (_go2\_env\_cfg.py:339_):

```python
"track_lin_vel_xy_exp": RewTerm(
    func=mdp.track_lin_vel_xy_exp,
    weight=weights["track_lin_vel_xy_exp"],            # 1.5 no baseline
    params={"command_name": "base_velocity", "std": math.sqrt(0.25)},  # std = 0.5
),
```

**Implementação** (Isaac Lab, _rewards.py:297_):

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
*   `Δv = v_comando − v_real`, medido no **body frame** (frame do robô)
*   `std = 0.5`, então `std² = 0.25`
_"quanto mais perto do comando, mais perto de 1"_

<div class="video-pair">
  <div class="video-panel">
    <p><strong>tracklin_0p5</strong></p>
    <video controls preload="metadata">
      <source src="{{ '/videos/sweep_tracklin_0p5_policy-step-999424000.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>tracklin_1p0</strong></p>
    <video controls preload="metadata">
      <source src="{{ '/videos/sweep_tracklin_1p0_policy-step-499712000.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>tracklin_3p0</strong></p>
    <video controls preload="metadata">
      <source src="{{ '/videos/sweep_tracklin_3p0_policy-step-499712000.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>tracklin_5p0</strong></p>
    <video controls preload="metadata">
      <source src="{{ '/videos/sweep_tracklin_5p0_policy-step-499712000.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

| config | valor | razão vs baseline (1.5) | Link |
| ---| ---| ---| --- |
| `sweep_tracklin_0p5.yml` | 0.5 | ÷3 | [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/t8cqki1k?nw=nwuserimdudak) |
| `sweep_tracklin_1p0.yml` | 1.0 | ÷1.5 | [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/41txvwq2?nw=nwuserimdudak) |
| `sweep_tracklin_3p0.yml` | 3.0 | ×2 | [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/zga8mchg?nw=nwuserimdudak) |
| `sweep_tracklin_5p0.yml` | 5.0 | ×3 | [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/men7c53s?nw=nwuserimdudak) |

É até então a recompensa que mais impacta na tarefa de andar do robô. Valores muito baixos simplesmente não conseguem dar o incentivo certeiro de andar. Valores altos demais fazem com que alcançar a velocidade X seja muito superior que respeitar o restante das recompensas.
### track\_ang\_vel\_z\_exp
**Registro** (_go2\_env\_cfg.py:344_):

```cs
"track_ang_vel_z_exp": RewTerm(
    func=mdp.track_ang_vel_z_exp,
    weight=weights["track_ang_vel_z_exp"],            # 0.75 no baseline
    params={"command_name": "base_velocity", "std": math.sqrt(0.25)},  # std = 0.5
),
```

**Implementação** (Isaac Lab, _rewards.py:308_):

```python
def track_ang_vel_z_exp(env, std, command_name, asset_cfg):
    ang_vel_error = torch.square(command[:, 2] - root_ang_vel_b[:, 2])  # (Δω)²
    return torch.exp(-ang_vel_error / std**2)
```

`r = exp( −(Δω)² / std² ) = exp( −4 · (Δω)² )`
*   `Δω = ω_cmd − ω_real` = erro da **taxa de giro** (yaw), no body frame
*   `std = 0.5` (mesmo `math.sqrt(0.25)`, linha 347) → `std² = 0.25`
*   peso baseline = **0.75** (metade do `track_lin` 1.5)
_"quanto mais perto do comando, mais perto de 1"_

<div class="video-pair">
  <div class="video-panel">
    <p><strong>trackang_0p25</strong></p>
    <video controls preload="metadata">
      <source src="{{ '/videos/sweep_trackang_0p25_policy-step-999424000.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>trackang_0p5</strong></p>
    <video controls preload="metadata">
      <source src="{{ '/videos/sweep_trackang_0p5_policy-step-999424000.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>trackang_1p5</strong></p>
    <video controls preload="metadata">
      <source src="{{ '/videos/sweep_trackang_1p5_policy-step-999424000.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>trackang_3p0</strong></p>
    <video controls preload="metadata">
      <source src="{{ '/videos/sweep_trackang_3p0_policy-step-999424000.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

| config | valor | razão vs baseline (0.75) | links |
| ---| ---| ---| --- |
| `sweep_trackang_0p25.yml` | 0.25 | ÷3 | [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/fo3uec6e?nw=nwuserimdudak) |
| `sweep_trackang_0p5.yml` | 0.5 | ÷1.5 | [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/aadvv7fk?nw=nwuserimdudak) |
| `sweep_trackang_1p5.yml` | 1.5 | ×2 | [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/5c8bn5bw?nw=nwuserimdudak) |
| `sweep_trackang_3p0.yml` | 3.0 | ×4 | [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/hy621to7?nw=nwuserimdudak) |

As recompensas lin\_vel\_z e lin\_vel\_xy competem entre si.

![](https://t90132646619.p.clickup-attachments.com/t90132646619/486f81cf-8ebc-45dd-b2b6-89cd8e86d4e8/image.png)

![](https://t90132646619.p.clickup-attachments.com/t90132646619/8c2c00a6-fa07-4277-b8ab-a4d621d1fe54/image.png)

### lin\_vel\_z\_l2
**Registro** (_go2\_env\_cfg.py:349_):

```python
"lin_vel_z_l2": RewTerm(func=mdp.lin_vel_z_l2, weight=weights["lin_vel_z_l2"]),  # -2.0 no baseline
```

**Implementação** (Isaac Lab, rewards.py):

```python
def lin_vel_z_l2(env, asset_cfg):
    return torch.square(asset.data.root_lin_vel_b[:, 2])   # (vz)²
```

`penalidade = (v_z)²`
onde:
*   `v_z` = velocidade linear do tronco no eixo **vertical (z)**, no body frame
*   peso **−2.0** (negativo → é penalidade)
#### O que ela faz
Pune o tronco **subir e descer** (movimento vertical). É um kernel **L2** (quadrático), sem `exp`: cresce com o **quadrado** da velocidade vertical — então velocidade pequena custa pouco, mas velocidade grande custa **muito** (penaliza picos bruscos).
A ideia: numa caminhada boa, o tronco deve ficar numa altura **estável**, sem **quicar**. Toda vez que o robô pula, balança verticalmente ou bate no chão e ressalta, gera `v_z` ≠ 0 e é punido.

<div class="video-pair">
  <div class="video-panel">
    <p><strong>linvelz_5p0</strong></p>
    <video controls preload="metadata">
      <source src="{{ '/videos/sweep_linvelz_5p0_policy-step-999424000.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>linvelz_3p0</strong></p>
    <video controls preload="metadata">
      <source src="{{ '/videos/sweep_linvelz_3p0_policy-step-999424000.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>linvelz_1p0</strong></p>
    <video controls preload="metadata">
      <source src="{{ '/videos/sweep_linvelz_1p0_policy-step-999424000.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>linvelz_0p5</strong></p>
    <video controls preload="metadata">
      <source src="{{ '/videos/sweep_linvelz_0p5_policy-step-999424000.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

### ang\_vel\_xy\_l2
**Registro** (_go2\_env\_cfg.py:350_):

```python
"ang_vel_xy_l2": RewTerm(func=mdp.ang_vel_xy_l2, weight=weights["ang_vel_xy_l2"]),  # -0.05 no baseline
```

**Implementação** (Isaac Lab, _rewards.py:83_):

```python
def ang_vel_xy_l2(env, asset_cfg):
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)  # ω_roll² + ω_pitch²
```

`penalidade = ω_x² + ω_y²`
onde:
*   `ω_x, ω_y` = taxas de **roll** e **pitch** do tronco (body frame)
*   peso **−0.05** (negativo → penalidade)

Kernel **L2** (sem `exp`): pune o tronco **balançar/tombar** — cabecear pra frente/trás e rolar de lado. Cresce com o quadrado da velocidade angular, então oscilação pequena custa pouco e tombo brusco custa muito. Não pune o giro de **yaw** (eixo z) — esse é tarefa do `track_ang_vel_z_exp`.

### dof\_torques\_l2
**Registro** (_go2\_env\_cfg.py:351_):

```python
"dof_torques_l2": RewTerm(func=mdp.joint_torques_l2, weight=weights["dof_torques_l2"]),  # -0.0002 no baseline
```

**Implementação** (Isaac Lab, _rewards.py:136_):

```python
def joint_torques_l2(env, asset_cfg):
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)  # Σ τ²
```

`penalidade = Σ_junta τ²`
onde:
*   `τ` = torque aplicado em cada junta
*   peso **−0.0002**

Pune o **esforço** das juntas (soma dos torques ao quadrado). Incentiva movimento mais **econômico e suave**. Alto demais deixa o robô "frouxo" (não levanta direito / freia a aceleração); baixo demais libera torque e aparece tremor e gasto de energia.

### dof\_acc\_l2
**Registro** (_go2\_env\_cfg.py:352_):

```python
"dof_acc_l2": RewTerm(func=mdp.joint_acc_l2, weight=weights["dof_acc_l2"]),  # -2.5e-7 no baseline
```

**Implementação** (Isaac Lab, _rewards.py:163_):

```python
def joint_acc_l2(env, asset_cfg):
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)  # Σ q̈²
```

`penalidade = Σ_junta q̈²`
onde:
*   `q̈` = aceleração de cada junta
*   peso **−2.5e-7** (numericamente minúsculo de propósito: `q̈` é enorme)

Pune **mudanças bruscas** de velocidade articular → é o alvo direto do **tremor de alta frequência**. Alto deixa as juntas mais suaves porém com reação mais lenta; baixo permite movimento nervoso/tremido.

### action\_rate\_l2
**Registro** (_go2\_env\_cfg.py:353_):

```python
"action_rate_l2": RewTerm(func=mdp.action_rate_l2, weight=weights["action_rate_l2"]),  # -0.01 no baseline
```

**Implementação** (Isaac Lab, _rewards.py:245_):

```python
def action_rate_l2(env):
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
```

`penalidade = Σ (a_t − a_{t-1})²`
onde:
*   `a_t` = ação (alvo de posição) deste step; `a_{t-1}` = a do step anterior
*   peso **−0.01**

Suaviza a **saída da política** (não a física direta): pune comandos que pulam de um step pro outro. Diferente do `dof_acc_l2`, atua **antes** da dinâmica. Alto deixa a política menos responsiva; baixo permite ações oscilando passo-a-passo → tremor.

### feet\_air\_time
**Registro** (_go2\_env\_cfg.py:354_):

```python
"feet_air_time": RewTerm(
    func=feet_air_time,
    weight=weights["feet_air_time"],                  # 0.5 no baseline
    params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        "command_name": "base_velocity",
        "threshold": 0.5,
    },
),
```

**Implementação** (_go2\_mdp.py:287_):

```python
def feet_air_time(env, command_name, sensor_cfg, threshold):
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1  # zera se parado
    return reward
```

`r = Σ_pé (t_ar − threshold) · primeiro_contato`
onde:
*   `t_ar` = tempo que o pé ficou no ar até tocar; `threshold = 0.5 s`
*   só conta no **primeiro contato** (no step em que o pé pousa)
*   **zera quando o comando é ~0** (não premia bater o pé parado)
*   peso **+0.5** (é **recompensa**, positiva)

Incentiva **passadas marcadas** (pé fica mais de 0.5 s no ar) em vez de arrastar/pisar miúdo. Alto demais pode virar "marcha no lugar"/saltitar pra farmar tempo de ar; baixo demais → passada curta e arrasto.

### feet\_slide
**Registro** (_go2\_env\_cfg.py:401_):

```python
"feet_slide": RewTerm(
    func=feet_slide,
    weight=weights["feet_slide"],                     # -0.25 no baseline
    params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
    },
),
```

**Implementação** (_go2\_mdp.py:305_):

```python
def feet_slide(env, sensor_cfg, asset_cfg):
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]  # velocidade xy do pé
    return torch.sum(body_vel.norm(dim=-1) * contacts.float(), dim=1)
```

`penalidade = Σ_pé ‖v_xy^pé‖ · (em contato)`
onde:
*   `v_xy^pé` = velocidade horizontal do pé; só conta quando o pé está **em contato** (força > 1 N)
*   peso **−0.25**

Pune o pé **escorregar enquanto apoia** — é o fix estrutural do arrasto: torna o pé deslizando custoso independente da magnitude do comando. Alto "crava" o pé no chão (marcha limpa, porém pode enrijecer/frear); baixo permite arrastar → deriva e desgaste.

### undesired\_contacts
**Registro** (_go2\_env\_cfg.py:372_):

```python
"undesired_contacts": RewTerm(
    func=mdp.undesired_contacts,
    weight=weights["undesired_contacts"],             # -0.2 no baseline
    params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*thigh"),
        "threshold": 1.0,
    },
),
```

**Implementação** (Isaac Lab, _rewards.py:260_):

```python
def undesired_contacts(env, threshold, sensor_cfg):
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)  # nº de corpos monitorados em contato
```

`penalidade = Σ_corpo (‖F_contato‖ > threshold)`
onde:
*   corpos monitorados = **coxas** (`.*thigh`); `threshold = 1.0 N`
*   conta **quantos** desses corpos estão tocando o chão (0..N)
*   peso **−0.2**

Pune contato em partes que **não deveriam** tocar o chão (ex.: coxas). Força apoiar só nas **patas** (postura correta). Alto pode deixar o robô cauteloso/lento; baixo tolera contatos espúrios (se apoiar errado/"ajoelhar").

### illegal\_contact\_penalty
**Registro** (_go2\_env\_cfg.py:380_):

```python
"illegal_contact_penalty": RewTerm(
    func=mdp.is_terminated_term,
    weight=weights["illegal_contact_penalty"],        # -1.0 no baseline
    params={"term_keys": "illegal_contact"},
),
```

**Implementação** (Isaac Lab, `is_terminated_term`) — recompensa de **evento de término**: vale `1.0` no step em que o episódio termina por aquele termo (`0` caso contrário). O termo de término correspondente (_go2\_env\_cfg.py:452_):

```python
illegal_contact = DoneTerm(
    func=mdp.illegal_contact,
    params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
)
```

`penalidade = −1.0` **no step em que termina** por contato ilegal
onde:
*   "contato ilegal" = a **base** (tronco) tocar o chão com força > 1 N → encerra o episódio
*   sinal **esparso** (só no término), mais forte que o `undesired_contacts`

Define o quanto **deitar a base no chão** "dói". Alto demais → conservador/parado; baixo → ignora colisões da base.

<div class="video-pair">
  <div class="video-panel">
    <p><strong>illegal_0p25</strong></p>
    <video controls preload="metadata">
      <source src="{{ '/videos/sweep_illegal_0p25_policy-step-999424000.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>illegal_0p5</strong></p>
    <video controls preload="metadata">
      <source src="{{ '/videos/sweep_illegal_0p5_policy-step-999424000.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>illegal_2p0</strong></p>
    <video controls preload="metadata">
      <source src="{{ '/videos/sweep_illegal_2p0_policy-step-999424000.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>illegal_4p0</strong></p>
    <video controls preload="metadata">
      <source src="{{ '/videos/sweep_illegal_4p0_policy-step-999424000.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

| config | valor | razão vs baseline (-1.0) | comportamento | métricas |
| ---| ---| ---| ---| --- |
| `sweep_illegal_0p25.yml` | -0.25 | ÷4 | anda, igual ao baseline | [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/wie86kph) |
| `sweep_illegal_0p5.yml` | -0.5 | ÷2 | anda, igual ao baseline | [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/vj89knrv) |
| `sweep_illegal_2p0.yml` | -2.0 | ×2 | anda, mas pior — transição atrasa ~250M steps | [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/sveaysll) |
| `sweep_illegal_4p0.yml` | -4.0 | ×4 | **não anda** — fica no ótimo local "sobreviver parado" | [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/0sbtyyry) |

**Resultados do sweep — o efeito é assimétrico:**

*   **Abaixar é inócuo** (`-0.25`, `-0.5`): curvas praticamente idênticas ao baseline em todos os termos. Faz sentido: pra uma política que já anda bem, a base tocar o chão é evento **raro** — o termo quase não dispara, então o peso dele pouco importa nessa direção.
*   **Aumentar atrasa ou impede a transição pra andar.** A "transição de fase" (~500–600M no baseline, visível no `track_lin_vel_xy_exp` saturando em ~1.25, no `is_alive` caindo de 0.4998→0.4992 e no `episodic_acc_length` caindo de ~700→~430) acontece **~250M mais tarde** no `-2.0` (só ~800M) e **nunca acontece** no `-4.0` dentro de 1B steps.
*   **O `-4.0` trava no ótimo local covarde**: `track_lin_vel` preso em ~0.25, episódio colado no teto (~700 = timeout, nunca cai), `illegal_contact_penalty` que espetava até −0.4/step no início (peso ×4 amplifica cada término) zera de vez — a política elimina o risco **parando de tentar andar**. A `entropy` fica sistematicamente mais alta (~10–11 vs ~8) — nunca converge pra um gait — e o `dof_torques_l2` mais negativo mostra que gasta torque "se remexendo" sem sair do lugar.
*   **Quem anda paga um pedágio terminal**: depois da transição, os runs que andam acumulam `head_contact_penalty` (~−0.6/step no eval) e episódios mais curtos — andar rápido traz términos ocasionais. É exatamente esse pedágio que o `-4.0` torna caro demais: o custo esperado de tentar andar supera o ganho de tracking, e "ficar parado vivo" vence.
  
 ![Gráfico velocidade XY]({{ '/images/illegal_contact_penalty.png' | relative_url }})


**Conclusão — `illegal_contact_penalty`:** janela segura de **−0.25 a −1.0** (indiferente); a partir de **−2.0** começa a competir com a tarefa e em **−4.0** domina. O peso funciona como uma **barreira de energia** pra transição: aprender a andar exige atravessar uma fase intermediária desajeitada em que a base toca o chão com frequência — e é exatamente essa fase que a penalidade taxa. Quanto maior o peso, mais caro o "vale" entre parado-estável e andando-estável: o atraso da transição cresce com o peso (−2.0 → +250M) até um ponto em que o PPO nunca acha que vale a pena atravessar (−4.0). Mesma família de fenômeno do `is_alive`/`fall_penalty`: penalidade terminal forte não ensina a "não colidir andando" — ensina a **não andar**. Predição pros sweeps de `fall_penalty` e `head_contact_penalty`: mesmo padrão de atraso monotônico, com o `fall_8p0` como candidato a travar de vez.

### fall\_penalty
**Registro** (_go2\_env\_cfg.py:385_):

```python
"fall_penalty": RewTerm(
    func=mdp.is_terminated_term,
    weight=weights["fall_penalty"],                   # -2.0 no baseline
    params={"term_keys": "robot_fell"},
),
```

**Implementação** (`is_terminated_term`) — término `robot_fell` (_go2\_env\_cfg.py:456_):

```python
robot_fell = DoneTerm(func=robot_fell, params={"height_threshold": 0.1})
```

`penalidade = −2.0` **no step em que cai**
onde:
*   "cair" = altura da base abaixo de **0.1 m** → encerra o episódio
*   é a penalidade terminal **mais forte** do baseline

Define o quanto **cair** custa. Alto demais → cauteloso demais, trava num ótimo local "não arrisca"; baixo demais → aceita quedas em troca de velocidade. Compete com o `is_alive` (bônus de sobreviver).

### head\_contact\_penalty
**Registro** (_go2\_env\_cfg.py:390_):

```python
"head_contact_penalty": RewTerm(
    func=mdp.is_terminated_term,
    weight=weights["head_contact_penalty"],           # -1.0 no baseline
    params={"term_keys": "head_contact"},
),
```

**Implementação** (`is_terminated_term`) — término `head_contact` (_go2\_env\_cfg.py:454_):

```python
head_contact = DoneTerm(
    func=mdp.illegal_contact,
    params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Head.*"), "threshold": 1.0},
)
```

`penalidade = −1.0` **no step em que a cabeça toca o chão**
onde:
*   corpos `Head.*` com força de contato > 1 N → encerra o episódio
*   peso **−1.0**

Protege contra **mergulhar a frente do corpo** (cabeçada/tombo pra frente). Alto mantém a cabeça erguida (pode travar/cauteloso); baixo tolera abaixar demais a frente.

### flat\_orientation\_l2
**Registro** (_go2\_env\_cfg.py:399_):

```python
"flat_orientation_l2": RewTerm(func=mdp.flat_orientation_l2, weight=weights["flat_orientation_l2"]),  # -0.25 no baseline
```

**Implementação** (Isaac Lab, _rewards.py:90_):

```python
def flat_orientation_l2(env, asset_cfg):
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)  # g_x² + g_y²
```

`penalidade = g_x² + g_y²`
onde:
*   `(g_x, g_y)` = componentes **horizontais** da gravidade projetada no body frame; valem `0` com o tronco perfeitamente **nivelado** e crescem com a inclinação
*   peso **−0.25**

Pune o tronco **inclinado**. Alto força postura horizontal (estável, mas pode impedir inclinar pra acelerar/virar); baixo libera o balanço do corpo (mais ágil porém pode cabecear/tombar). Relaciona-se com `head_contact_penalty` e `ang_vel_xy_l2`.

## hiperparâmetros
### learning\_rate
**Otimizador:**
É o tamanho do passo do Adam que atualiza **a política e o critic juntos** (mesma rede/otimizador).

```python
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
```

**Annealing:**
Com `anneal_lr: true`, a LR **decai linearmente até 0** ao longo do `total_timesteps`

```python
if args.anneal_lr:
    frac = 1.0 - (iteration - 1.0) / num_iterations
    lrnow = frac * args.learning_rate
    optimizer.param_groups[0]["lr"] = lrnow
```

Treinos:
Sweep lr 0003 : [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/w6qzxu82?nw=nwuserimdudak) | Sweep lr 003 : [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/0d0cxqse?nw=nwuserimdudak)

<div class="video-pair">
  <div class="video-panel">
    <p><strong>lr_0003</strong></p>
    <video controls preload="metadata">
      <source src="{{ '/videos/sweep_lr_0003_policy-step-999424000.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>lr_003</strong></p>
    <video controls preload="metadata">
      <source src="{{ '/videos/sweep_lr_003_policy-step-999424000.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

#### Como impacta o aprendizado?
*   Controla **quão rápido** a política muda a cada update. Não muda o gradiente (a direção), só a **magnitude do passo**.
*   Como esperado, diminuir o LR fez com que o aprendizado demorasse um pouco mais para convergir, mas ainda chega no mesmo local que as outras configs.![](https://t90132646619.p.clickup-attachments.com/t90132646619/b7a9d1f5-0a8e-4e5e-8142-7e9c7399aa69/image.png)

####   

| param | atual | efeito de aumentar | sintoma se alto demais | se baixo demais |
| ---| ---| ---| ---| --- |
| `track_lin_vel_xy_exp` | 1.5 | foco em seguir velocidade | ignora estabilidade/suavidade | não anda no comando |
| `track_ang_vel_z_exp` | 0.75 | foco no yaw | sacrifica linear | não vira |
| `is_alive` | 0.5 | "só sobreviver" vale mais | ótimo local: anda aleatório só pra não cair | cai fácil |
| `feet_air_time` | 0.5 | passos mais marcados | pula/exagera a passada | arrasta/escorrega |
| `feet_slide` | \-0.25 | pune arrastar pé | trava o pé, anda rígido | arrasta |
| `action_rate_l2` | \-0.01 | suaviza ações | robô lento/cauteloso | ações bruscas, tremor |
| `dof_torques_l2` | \-0.0002 | economiza torque | fraco, não levanta direito | gasta energia, brusco |
| `flat_orientation_l2` | \-0.25 | tronco nivelado | não inclina pra acelerar | tomba/senta |
| `fall_penalty` | \-2.0 | medo de cair | cauteloso demais, não arrisca | cai sem custo |

## PPO — hiperparâmetros

| param | atual | range são | alto demais | baixo demais | olhar |
| ---| ---| ---| ---| ---| --- |
| `learning_rate` | 0.001 | 1e-4 – 3e-3 | KL dispara, entropia explode, colapso | não aprende / lento | `old_approx_kl`, entropy |
| `num_steps` | 100 | 24 – 200 | dados velhos, update lento | advantage ruidoso, instável | `explained_variance` |
| `update_epochs` | 5 | 3 – 8 | overfit no batch, KL deriva | subaproveita os dados | `old_approx_kl` |
| `ent_coef` | 0.008 | 0 – 0.012 | entropia só sobe, política aleatória | converge cedo demais (trava) | curva de `entropy` |
| `num_minibatches` | 16 | 4 – 32 | gradiente ruidoso | menos updates | `policy_loss` |
| `gamma` | 0.99 | 0.98 – 0.995 | valor difícil de aprender | míope, não planeja passada | `value_loss` |
| `gae_lambda` | 0.95 | 0.9 – 0.97 | advantage com variância alta | enviesado | `explained_variance` |
| `target_kl` | null | null ou 0.01–0.02 | (setar) corta updates ruins | — | guarda contra blowup |
