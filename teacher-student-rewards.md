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

Esses vídeos usam a config base acima, com `track_lin_vel_xy_exp: 1.5`. Eles servem como referência antes de comparar os sweeps da recompensa.

**Teacher**

<video controls preload="metadata" width="100%">
  <source src="{{ '/videos/policy-step-190021632-teacher.mp4' | relative_url }}" type="video/mp4">
</video>

**Student**

<video controls preload="metadata" width="100%">
  <source src="{{ '/videos/policy-step-190021632-student.mp4' | relative_url }}" type="video/mp4">
</video>

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

No Teacher–Student, essa recompensa tem dois efeitos. Primeiro, ela força o **teacher** a aprender uma política que segue velocidade linear. Depois, o **student** tenta reproduzir esse comportamento usando histórico de observações e o latente de adaptação.

## Sweep `track_lin_vel_xy_exp`

### `track_lin_vel_xy_exp = 0.5`

Com peso `0.5`, o tracking linear fica fraco em relação aos termos de estabilidade e penalidades. Esse teste indica se o Teacher–Student ainda aprende locomoção útil quando seguir velocidade não é tão vantajoso.

**Teacher**

<video controls preload="metadata" width="100%">
  <source src="{{ '/videos/policy-step-299925504-teacher-tracklin_0p5.mp4' | relative_url }}" type="video/mp4">
</video>

**Student**

O arquivo `policy-step-299925504-student-tracklin_0p5.mp4` está corrompido/incompleto (`moov atom not found`), então este ponto ainda não permite comparação visual direta entre teacher e student.

Leitura: por enquanto, o peso `0.5` fica inconclusivo para o student. Pela função da recompensa, a hipótese é que esse valor dê pouco incentivo para tracking linear, gerando uma política mais conservadora ou menos direcionada ao comando.

### `track_lin_vel_xy_exp = 1.0`

Com peso `1.0`, o tracking linear ainda fica abaixo do baseline `1.5`, mas já compete melhor com os termos de estabilidade. É uma redução moderada: a política deve seguir o comando, mas sem tornar velocidade o único objetivo dominante.

**Teacher**

<video controls preload="metadata" width="100%">
  <source src="{{ '/videos/policy-step-280068096-teacher-tracklin-1p0.mp4' | relative_url }}" type="video/mp4">
</video>

**Student**

<video controls preload="metadata" width="100%">
  <source src="{{ '/videos/policy-step-280068096-student-tracklin-1p0.mp4' | relative_url }}" type="video/mp4">
</video>

Leitura: este peso ajuda a verificar se o student mantém uma marcha estável quando o teacher não está superpriorizando velocidade. Se o tracking ficar fraco, mas a marcha estiver limpa, o valor pode ser baixo demais para a tarefa principal.

### `track_lin_vel_xy_exp = 3.0`

Com peso `3.0`, o tracking linear fica duas vezes maior que o baseline `1.5`. A política recebe incentivo forte para seguir a velocidade comandada, mas ainda não tão extremo quanto `5.0`.

**Teacher**

<video controls preload="metadata" width="100%">
  <source src="{{ '/videos/policy-step-280068096-teacher-tracklin-3p0.mp4' | relative_url }}" type="video/mp4">
</video>

**Student**

<video controls preload="metadata" width="100%">
  <source src="{{ '/videos/policy-step-280068096-student-tracklin-3p0.mp4' | relative_url }}" type="video/mp4">
</video>

Leitura: este é o principal candidato para comparação com o baseline. Se o teacher melhora o tracking sem perder estabilidade, e o student reproduz a marcha sem colapsar, `3.0` pode ser um bom ajuste. O risco é começar a sacrificar contato e suavidade para perseguir velocidade.

### `track_lin_vel_xy_exp = 5.0`

Com peso `5.0`, seguir a velocidade comandada vira um objetivo muito dominante. Esse teste mostra o limite superior: até onde aumentar tracking ajuda antes de prejudicar a qualidade da marcha.

**Teacher**

<video controls preload="metadata" width="100%">
  <source src="{{ '/videos/policy-step-280068096-teacher-tracklin-5p0.mp4' | relative_url }}" type="video/mp4">
</video>

**Student**

<video controls preload="metadata" width="100%">
  <source src="{{ '/videos/policy-step-280068096-student-tracklin-5p0.mp4' | relative_url }}" type="video/mp4">
</video>

Leitura: esse peso deve ser analisado com cuidado. Mesmo que o robô pareça responder melhor ao comando, a recompensa pode deslocar a política para uma solução menos estável. No Teacher–Student, isso pesa mais porque o student pode amplificar problemas do teacher quando precisa adaptar o comportamento a partir de observações parciais.

## Resumo dos testes

| config | valor | razão vs baseline TS (`1.5`) | vídeos |
| --- | --- | --- | --- |
| `teacher/student-tracklin_0p5` | `0.5` | ÷3 | teacher disponível; student inválido |
| `teacher/student-tracklin-1p0` | `1.0` | ÷1.5 | teacher e student disponíveis |
| `teacher/student-tracklin-3p0` | `3.0` | ×2 | teacher e student disponíveis |
| `teacher/student-tracklin-5p0` | `5.0` | ×3.33 | teacher e student disponíveis |

## Interpretação geral

`track_lin_vel_xy_exp` continua sendo uma recompensa central para fazer o robô andar no Teacher–Student. Ela define a pressão para seguir o comando linear, enquanto os outros termos seguram estabilidade, postura, contato e suavidade.

No baseline Teacher–Student, o peso é `1.5`. Os testes abaixo desse valor (`0.5` e `1.0`) ajudam a identificar se o robô consegue manter marcha com menos pressão de tracking. Os testes acima (`3.0` e `5.0`) mostram se aumentar o peso melhora a resposta ao comando ou se começa a gerar uma política que prioriza velocidade demais.

A análise final deve comparar teacher e student separadamente:

* no **teacher**, observar se a recompensa melhora o comportamento alvo;
* no **student**, observar se a política transferida preserva esse comportamento;
* se o teacher melhora mas o student piora, o problema pode estar na transferência/adaptação;
* se teacher e student pioram juntos, o peso provavelmente está deslocando a otimização para um comportamento ruim.

## Critérios de comparação

Ao comparar os vídeos, olhar principalmente:

| critério | o que observar |
| --- | --- |
| tracking linear | se o robô acompanha o comando de velocidade sem hesitar |
| estabilidade | se o tronco fica controlado e não quica demais |
| contato | se há arrasto, escorregamento ou perda de ritmo nas patas |
| suavidade | se a marcha é contínua ou cheia de correções bruscas |
| transferência | se o student parece reproduzir o teacher ou degrada o comportamento |
| robustez | se aumentar o peso melhora tracking sem piorar quedas, postura ou contato |

Conclusão provisória: `3.0` é o ponto mais interessante para investigar depois do baseline, porque aumenta bastante o incentivo de tracking sem ir diretamente para o caso extremo de `5.0`. O valor `5.0` deve ser tratado como teste de limite. O valor `0.5` precisa do vídeo válido do student antes de qualquer conclusão.
