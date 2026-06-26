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
[https://t90132646619.p.clickup-attachments.com/t90132646619/608154b7-9a02-4217-941e-f4fb7f41293a/config\_policy-step-999424000.mp4?view=open](https://t90132646619.p.clickup-attachments.com/t90132646619/608154b7-9a02-4217-941e-f4fb7f41293a/config_policy-step-999424000.mp4?view=open)
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
Sweep isalive\_0: [https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/1uaj7uq0?nw=nwuserimdudak](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/1uaj7uq0?nw=nwuserimdudak) | Sweep isalive\_1: [https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/sivema1e?nw=nwuserimdudak](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/sivema1e?nw=nwuserimdudak)

[https://t90132646619.p.clickup-attachments.com/t90132646619/fd02ba64-ffe5-4e79-a39c-4494ce151b19/sweep\_isalive\_0\_policy-step-999424000.mp4?view=open](https://t90132646619.p.clickup-attachments.com/t90132646619/fd02ba64-ffe5-4e79-a39c-4494ce151b19/sweep_isalive_0_policy-step-999424000.mp4?view=open)

[https://t90132646619.p.clickup-attachments.com/t90132646619/6ef146b5-2695-409e-b7c3-7bb88a180f5d/sweep\_isalive\_1p0\_policy-step-999424000.mp4policy-step-999424000.mp4?view=open](https://t90132646619.p.clickup-attachments.com/t90132646619/6ef146b5-2695-409e-b7c3-7bb88a180f5d/sweep_isalive_1p0_policy-step-999424000.mp4policy-step-999424000.mp4?view=open)

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

[https://t90132646619.p.clickup-attachments.com/t90132646619/23f59132-88e8-44b7-92dd-547ee946fe0c/sweep\_tracklin\_0p5\_policy-step-999424000.mp4policy-step-999424000.mp4?view=open](https://t90132646619.p.clickup-attachments.com/t90132646619/23f59132-88e8-44b7-92dd-547ee946fe0c/sweep_tracklin_0p5_policy-step-999424000.mp4policy-step-999424000.mp4?view=open)

[https://t90132646619.p.clickup-attachments.com/t90132646619/4db7d72d-e962-4ff3-a588-6fac9af3468f/sweep\_tracklin\_1p0\_policy-step-499712000.mp4policy-step-999424000.mp4?view=open](https://t90132646619.p.clickup-attachments.com/t90132646619/4db7d72d-e962-4ff3-a588-6fac9af3468f/sweep_tracklin_1p0_policy-step-499712000.mp4policy-step-999424000.mp4?view=open)

[https://t90132646619.p.clickup-attachments.com/t90132646619/aa4d2e1b-6017-44cb-82de-b9d749e3f6e5/sweep\_tracklin\_3p0\_policy-step-499712000.mp4policy-step-999424000.mp4?view=open](https://t90132646619.p.clickup-attachments.com/t90132646619/aa4d2e1b-6017-44cb-82de-b9d749e3f6e5/sweep_tracklin_3p0_policy-step-499712000.mp4policy-step-999424000.mp4?view=open)

[https://t90132646619.p.clickup-attachments.com/t90132646619/400d012a-568c-4bd8-b5a9-364974767401/sweep\_tracklin\_5p0\_policy-step-499712000.mp4policy-step-999424000.mp4?view=open](https://t90132646619.p.clickup-attachments.com/t90132646619/400d012a-568c-4bd8-b5a9-364974767401/sweep_tracklin_5p0_policy-step-499712000.mp4policy-step-999424000.mp4?view=open)

| config | valor | razão vs baseline (1.5) | Link |
| ---| ---| ---| --- |
| `sweep_tracklin_0p5.yml` | 0.5 | ÷3 | [https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/t8cqki1k?nw=nwuserimdudak](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/t8cqki1k?nw=nwuserimdudak) |
| `sweep_tracklin_1p0.yml` | 1.0 | ÷1.5 | [https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/41txvwq2?nw=nwuserimdudak](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/41txvwq2?nw=nwuserimdudak) |
| `sweep_tracklin_3p0.yml` | 3.0 | ×2 | [https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/zga8mchg?nw=nwuserimdudak](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/zga8mchg?nw=nwuserimdudak) |
| `sweep_tracklin_5p0.yml` | 5.0 | ×3 | [https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/men7c53s?nw=nwuserimdudak](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/men7c53s?nw=nwuserimdudak) |

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

[https://t90132646619.p.clickup-attachments.com/t90132646619/d62bbedd-2347-44d1-a188-b42fa84007e7/sweep\_trackang\_0p25\_policy-step-999424000.mp4policy-step-999424000.mp4?view=open](https://t90132646619.p.clickup-attachments.com/t90132646619/d62bbedd-2347-44d1-a188-b42fa84007e7/sweep_trackang_0p25_policy-step-999424000.mp4policy-step-999424000.mp4?view=open)

[https://t90132646619.p.clickup-attachments.com/t90132646619/edaab1be-6a17-4620-be57-66fa8a0c0d63/sweep\_trackang\_0p5\_policy-step-999424000.mp4?view=open](https://t90132646619.p.clickup-attachments.com/t90132646619/edaab1be-6a17-4620-be57-66fa8a0c0d63/sweep_trackang_0p5_policy-step-999424000.mp4?view=open)

[https://t90132646619.p.clickup-attachments.com/t90132646619/33f26c8c-c80a-4a68-ac24-d188fc5e8730/sweep\_trackang\_1p5\_policy-step-999424000.mp4policy-step-999424000.mp4?view=open](https://t90132646619.p.clickup-attachments.com/t90132646619/33f26c8c-c80a-4a68-ac24-d188fc5e8730/sweep_trackang_1p5_policy-step-999424000.mp4policy-step-999424000.mp4?view=open)

[https://t90132646619.p.clickup-attachments.com/t90132646619/f2eeae48-d206-4b49-b70f-9b53ba072ec3/sweep\_trackang\_3p0\_policy-step-999424000.mp4policy-step-999424000.mp4?view=open](https://t90132646619.p.clickup-attachments.com/t90132646619/f2eeae48-d206-4b49-b70f-9b53ba072ec3/sweep_trackang_3p0_policy-step-999424000.mp4policy-step-999424000.mp4?view=open)

| config | valor | razão vs baseline (0.75) | links |
| ---| ---| ---| --- |
| `sweep_trackang_0p25.yml` | 0.25 | ÷3 | [https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/fo3uec6e?nw=nwuserimdudak](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/fo3uec6e?nw=nwuserimdudak) |
| `sweep_trackang_0p5.yml` | 0.5 | ÷1.5 | [https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/aadvv7fk?nw=nwuserimdudak](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/aadvv7fk?nw=nwuserimdudak) |
| `sweep_trackang_1p5.yml` | 1.5 | ×2 | [https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/5c8bn5bw?nw=nwuserimdudak](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/5c8bn5bw?nw=nwuserimdudak) |
| `sweep_trackang_3p0.yml` | 3.0 | ×4 | [https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/hy621to7?nw=nwuserimdudak](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/hy621to7?nw=nwuserimdudak) |

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

[https://t90132646619.p.clickup-attachments.com/t90132646619/744e6e27-74b5-4745-9fbc-f35e5a052553/sweep\_linvelz\_5p0\_policy-step-999424000.mp4?view=open](https://t90132646619.p.clickup-attachments.com/t90132646619/744e6e27-74b5-4745-9fbc-f35e5a052553/sweep_linvelz_5p0_policy-step-999424000.mp4?view=open)

[sweep\_linvelz\_3p0\_policy-step-999424000.mp4](undefined)

[https://t90132646619.p.clickup-attachments.com/t90132646619/5d4f896c-78e4-491c-9841-be4ee2a841b7/sweep\_linvelz\_1p0\_policy-step-999424000.mp4?view=open](https://t90132646619.p.clickup-attachments.com/t90132646619/5d4f896c-78e4-491c-9841-be4ee2a841b7/sweep_linvelz_1p0_policy-step-999424000.mp4?view=open)[https://t90132646619.p.clickup-attachments.com/t90132646619/91ad907c-f671-4941-b8a7-34f7542e5b99/sweep\_linvelz\_0p5\_policy-step-999424000.mp4?view=open](https://t90132646619.p.clickup-attachments.com/t90132646619/91ad907c-f671-4941-b8a7-34f7542e5b99/sweep_linvelz_0p5_policy-step-999424000.mp4?view=open)
You don't have access to this Doc
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
Sweep lr 0003 : [https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/w6qzxu82?nw=nwuserimdudak](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/w6qzxu82?nw=nwuserimdudak) | Sweep lr 003 : [https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/0d0cxqse?nw=nwuserimdudak](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/0d0cxqse?nw=nwuserimdudak)

[https://t90132646619.p.clickup-attachments.com/t90132646619/1854d008-80ee-44f8-8b40-94a82705eb65/sweep\_lr\_0003\_policy-step-999424000.mp4?view=open](https://t90132646619.p.clickup-attachments.com/t90132646619/1854d008-80ee-44f8-8b40-94a82705eb65/sweep_lr_0003_policy-step-999424000.mp4?view=open)

[https://t90132646619.p.clickup-attachments.com/t90132646619/599cd032-a4fd-4aba-a304-a7b3d46106b5/sweep\_lr\_003\_policy-step-999424000.mp4policy-step-999424000.mp4?view=open](https://t90132646619.p.clickup-attachments.com/t90132646619/599cd032-a4fd-4aba-a304-a7b3d46106b5/sweep_lr_003_policy-step-999424000.mp4policy-step-999424000.mp4?view=open)

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
