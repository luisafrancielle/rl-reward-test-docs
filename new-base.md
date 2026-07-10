---
layout: default
title: Rewards · New Base
section: new-base
kicker: NEW BASE
description: Comparativo de Configurações de Recompensa (Apenas Parâmetros Modificados).
permalink: /new-base/
---

# Comparativo de Configurações de Recompensa (Apenas Parâmetros Modificados)

Esta tabela apresenta apenas as linhas/parâmetros de recompensa (rewards) que sofreram alguma modificação em relação ao arquivo de baseline.

<div class="video-panel">
  <p><strong>Baseline (`flat_vel_fwd_base.yml`)</strong></p>
  <video controls preload="metadata">
    <source src="{{ '/videos/new-base_policy-step-699924480.mp4' | relative_url }}" type="video/mp4">
  </video>
</div>

| Parâmetro Modificado | Baseline | Base 2 | Base 3 | Base 4 | Base 5 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Arquivo de Config** | `flat_vel_fwd_base.yml` | `flat_vel_fwd_base_2.yml` | `flat_vel_fwd_base_3.yml` | `flat_vel_fwd_base_4.yml` | `flat_vel_fwd_base_5.yml` |
| `track_ang_vel_z_exp` | 0.75 | 0.75 | 0.50 | 0.50 | 0.75 |
| `lin_vel_z_l2` | -2.0 | -3.0 | -3.0 | -3.0 | -2.0 |
| `action_rate_l2` | -0.01 | -0.01 | -0.01 | -0.005 | -0.01 |
| `flat_orientation_l2` | -0.25 | -0.125 | -0.125 | -0.125 | -1.0 |
| `feet_air_time` | 0.5 | 0.25 | 0.25 | 0.25 | 0.5 |
| `feet_slide` | -0.25 | -0.20 | -0.20 | -0.20 | -0.25 |
| **Link do WandB** | [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/bbeow45c?nw=nwuserimdudak) | [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/17nhcywi?nw=nwuserimdudak) | [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/ac9ktch4?nw=nwuserimdudak) | [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/sj5llh1c?nw=nwuserimdudak) | [wandb](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/oa1cgh7j?nw=nwuserimdudak) |

### Vídeos das Versões Modificadas

<div class="video-pair">
  <div class="video-panel">
    <p><strong>flat_vel_fwd_base_2.yml</strong></p>
    <video controls preload="metadata">
      <source src="{{ '/videos/new-base-2_policy-step-699924480.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>flat_vel_fwd_base_3.yml</strong></p>
    <video controls preload="metadata">
      <source src="{{ '/videos/new-base-3_policy-step-699924480.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>flat_vel_fwd_base_4.yml</strong></p>
    <video controls preload="metadata">
      <source src="{{ '/videos/new-base-4_policy-step-699924480.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
  <div class="video-panel">
    <p><strong>flat_vel_fwd_base_5.yml</strong></p>
    <video controls preload="metadata">
      <source src="{{ '/videos/new-base-5_policy-step-699924480.mp4' | relative_url }}" type="video/mp4">
    </video>
  </div>
</div>

---

## Detalhamento das Mudanças e Raciocínio (Baseado nos Comentários)

### 1. Mudanças no [`flat_vel_fwd_base_2.yml`](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/17nhcywi?nw=nwuserimdudak)

* **`lin_vel_z_l2`**: Reduzido de `-2.0` para `-3.0`.
  * **Raciocínio**: *"quanto mais negativo menor o bouncing of the robot"* (minimiza o balanço/oscilação vertical do robô).
* **`flat_orientation_l2`**: Ajustado de `-0.25` para `-0.125`.
* **`feet_air_time`**: Reduzido de `0.5` para `0.25`.
  * **Raciocínio**: Com base nos experimentos anteriores, pode ser menor, o seu efeito direto é diminuir o tempo da pata no ar, comportamento que queremos corrigir do baseline.
* **`feet_slide`**: Diminui de `-0.25` (mais punitivo) para `-0.20` (menos punitivo).
  * **Raciocínio**:  Com base nos experimentos anteriores, ainda estava no limiar de exploração e diminuir sutilmente faz com que o andar ganhe mais liberdade.

### 2. Mudanças no [`flat_vel_fwd_base_3.yml`](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/ac9ktch4?nw=nwuserimdudak)

* Mantém as alterações da versão 2 (`lin_vel_z_l2`, `flat_orientation_l2`, `feet_air_time` e `feet_slide`).
* **`track_ang_vel_z_exp`**: Reduzido de `0.75` para `0.50`.
  * **Raciocínio**: Com base nos experimentos anteriores, diminuir sutilmente o peso desse componente ajuda a melhorar a qualidade do andar.

### 3. Mudanças no [`flat_vel_fwd_base_4.yml`](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/sj5llh1c?nw=nwuserimdudak)

* Mantém as alterações da versão 3.
* **`action_rate_l2`**: Ajustado de `-0.01` para `-0.005`.
  * **Raciocínio**: Com base nos experimentos anteriores, reduzir o componente diminui a punição na taxa de variação de ação para evitar tremeliques ou travamento/instabilidade nas ações.

### 4. Mudanças no [`flat_vel_fwd_base_5.yml`](https://wandb.ai/imdudak-federal-university-of-goi-s/Akcit-RL/runs/oa1cgh7j?nw=nwuserimdudak)

* Parte diretamente do Baseline (não herda as modificações das versões 2, 3 e 4).
* **`flat_orientation_l2`**: Aumentado consideravelmente de `-0.25` para `-1.0` (penalidade muito mais rigorosa para orientações não-placas).
  * **Raciocínio**: Nos experimentos anteriores aumentar essa variável produziu um andar bem natural, quis testar aqui também.
