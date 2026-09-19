---
type: overview
tags: [open-source, curated-index, reinforcement-learning, imitation-learning, vla, wechat-curator, locomotion]
status: complete
updated: 2026-09-19
related:
  - ../entities/rsl-rl.md
  - ../entities/tienkung-lab.md
  - ../entities/unitree-rl-gym.md
  - ../entities/aloha.md
  - ../entities/paper-act.md
  - ../entities/paper-diffusion-policy.md
  - ../entities/openvla.md
  - ../entities/paper-octo.md
  - ../entities/lingbot-world.md
  - ../entities/lerobot.md
  - ../entities/pinocchio.md
  - ../entities/acados.md
  - ../entities/mujoco-menagerie.md
  - ../entities/unitree-mujoco.md
  - ../entities/awesome-legged-locomotion-learning.md
  - ../entities/awesome-physical-ai-natnew.md
  - ../entities/paper-amp-survey-01-amp.md
  - ../tasks/locomotion.md
sources:
  - ../../sources/blogs/wechat_robot_yanfa_opensource_algorithms_compendium.md
  - ../../sources/raw/wechat_robot_yanfa_opensource_algorithms_2026.md
summary: "微信公众号「机器人研发工程师」16 项开源算法/框架索引：每项独立 wiki 详情节点 + 经 2026-09-19 核查的可点击 GitHub 链接（文内失效链已校正）。"
---

# 开源机器人算法大全（微信策展）— 索引

## 一句话定义

本页把 [微信公众号清单](https://mp.weixin.qq.com/s/eXDsk8svhmLUkzMVlXd-qw) 中的 **16 项开源资源**拆成 **独立、不重复** 的 wiki 详情节点，并在此集中给出 **2026-09-19 核查通过** 的 GitHub 入口（原文部分链接已 404 或 org 笔误，见「链接校正」列）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 足式 PPO 步态主线 |
| IL | Imitation Learning | ALOHA / ACT / Diffusion Policy |
| VLA | Vision-Language-Action | OpenVLA / Octo / LingBot 家族 |
| MPC | Model Predictive Control | acados 求解栈 |
| MJCF | MuJoCo XML Format | Menagerie 资产格式 |

## 16 项一览（详情节点 + 可用链接）

| # | 分区 | 详情节点 | 核查后 GitHub / 入口 | 链接校正 |
|---|------|----------|----------------------|----------|
| 1 | 足式 RL | [RSL-RL](../entities/rsl-rl.md) | [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl) | ✓ 配套 [legged_gym](../entities/legged-gym.md) |
| 2 | 足式 RL | [AMP（论文实体）](../entities/paper-amp-survey-01-amp.md) | 论文方法；工程实现见 [AMP_for_hardware](../entities/amp-for-hardware.md) | 原文 `facebookresearch/amp` **404** |
| 3 | 足式 RL | [TienKung-Lab](../entities/tienkung-lab.md) | [Open-X-Humanoid/TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab) | ✓ |
| 4 | 足式 RL | [unitree_rl_gym](../entities/unitree-rl-gym.md) | [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) | ✓ |
| 5 | 模仿学习 | [ALOHA](../entities/aloha.md) + [ACT](../entities/paper-act.md) | [tonyzhaozh/aloha](https://github.com/tonyzhaozh/aloha) · [tonyzhaozh/act](https://github.com/tonyzhaozh/act) | 原文 `StanfordVL/ALOHA` **404** |
| 6 | 模仿学习 | [Diffusion Policy](../entities/paper-diffusion-policy.md) | [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) | 原文 org `real-strawberry` **404** |
| 7 | VLA | [OpenVLA](../entities/openvla.md) | [openvla/openvla](https://github.com/openvla/openvla) | ✓ |
| 8 | VLA | [Octo](../entities/paper-octo.md) | [octo-models/octo](https://github.com/octo-models/octo) | ✓ |
| 9 | VLA / WM | [LingBot-World](../entities/lingbot-world.md) | [robbyant/lingbot-world](https://github.com/robbyant/lingbot-world) | 原文 `antgroup/lingbot` **404** |
| 10 | 框架 | [LeRobot](../entities/lerobot.md) | [huggingface/lerobot](https://github.com/huggingface/lerobot) | ✓ |
| 11 | 动力学 | [Pinocchio](../entities/pinocchio.md) | [stack-of-tasks/pinocchio](https://github.com/stack-of-tasks/pinocchio) | ✓ |
| 12 | 动力学 | [acados](../entities/acados.md) | [acados/acados](https://github.com/acados/acados) | ✓ |
| 13 | MuJoCo 资产 | [MuJoCo Menagerie](../entities/mujoco-menagerie.md) | [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) | ✓ |
| 14 | MuJoCo 仿真 | [unitree_mujoco](../entities/unitree-mujoco.md) | [unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) | ✓ |
| 15 | Awesome | [awesome-legged-locomotion-learning](../entities/awesome-legged-locomotion-learning.md) | [gaiyi7788/awesome-legged-locomotion-learning](https://github.com/gaiyi7788/awesome-legged-locomotion-learning) | ✓ |
| 16 | Awesome | [awesome-physical-ai（natnew）](../entities/awesome-physical-ai-natnew.md) | [natnew/awesome-physical-ai](https://github.com/natnew/awesome-physical-ai) | ✓ |

## 文内推荐上手顺序（保留原意）

1. [RSL-RL](../entities/rsl-rl.md) + [unitree_mujoco](../entities/unitree-mujoco.md) — PPO 步态闭环  
2. [LeRobot](../entities/lerobot.md) + [ALOHA](../entities/aloha.md)/[ACT](../entities/paper-act.md) — 模仿学习流水线  
3. [Pinocchio](../entities/pinocchio.md) — 运动学 / 雅可比  
4. [Octo](../entities/paper-octo.md) 推理 → 再 [OpenVLA](../entities/openvla.md) 微调  

## 常见误区

- **误区：微信文内 GitHub 链永远有效。** 至少 AMP、ALOHA、Diffusion Policy、LingBot 四条需按上表校正。
- **误区：TienKung-Lab = 天工 URDF 页。** 训练框架见 [TienKung-Lab](../entities/tienkung-lab.md)；本体见 [天工开源](../entities/tienkung-humanoid-open-source.md)。
- **误区：MuJoCo Menagerie = MuJoCo 引擎。** 引擎见 [mujoco](../entities/mujoco.md)；Menagerie 仅为 **MJCF 资产库**。

## 关联页面

- [Locomotion](../tasks/locomotion.md)
- [Imitation Learning](../methods/imitation-learning.md)
- [VLA](../methods/vla.md)

## 参考来源

- [wechat_robot_yanfa_opensource_algorithms_compendium.md](../../sources/blogs/wechat_robot_yanfa_opensource_algorithms_compendium.md)
- [wechat_robot_yanfa_opensource_algorithms_2026.md](../../sources/raw/wechat_robot_yanfa_opensource_algorithms_2026.md)

## 推荐继续阅读

- 原文：[微信公众号](https://mp.weixin.qq.com/s/eXDsk8svhmLUkzMVlXd-qw)
