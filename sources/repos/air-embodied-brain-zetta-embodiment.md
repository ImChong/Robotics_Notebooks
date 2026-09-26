# Zetta-Embodiment（air-embodied-brain/Zetta-Embodiment）

- **URL：** <https://github.com/air-embodied-brain/Zetta-Embodiment>
- **组织：** air-embodied-brain
- **Stars：** ~1.2k+（2026-09-26 量级）
- **关联项目页：** [zetta-air-embodied-brain](../sites/zetta-air-embodied-brain.md)
- **关联论文：** [zetta_arxiv_2608_16590](../papers/zetta_arxiv_2608_16590.md)

## 一句话说明

**Zetta ζ** 官方实现：闭环 **critic/recovery** harness 自进化协议、**Rollout Runtime（Z-Infra）**、LIBERO-Pro（Pi0.5）/ RoboCasa（GR00T）/ RoboTwin / ManiSkill / Genie Sim 等 env 集成；campaign 目录含 manifest、cluster、diagnose、gate、promoted 等工件。

## 开源状态（2026-09-26）

**已开源**（代码与协议；VLA 权重、sim 资产、视频与 API 凭证 **不入库**）。

### README 入口摘要

| 路径 | 用途 |
|------|------|
| `zetta/evolution/` | 进化 manifest、聚类、Stage1/2、gate、promotion |
| `rollout_runtime/` | Gateway、EnvWorker/RolloutWorker、多 backend |
| `robots/libero|robocasa|robotwin/` | Role1 / Critic / Recovery / tools |
| `scripts/evolution/` | campaign 准备与 worker |
| `scripts/deployment/` | `install_vla_env.sh`（libero-pro / robocasa 分轨）、Docker |

**Evolution Protocol（摘要）：** 50 dev rollouts → Failure Cluster → Stage1 Diagnose → Stage2 Critic-Recovery 候选 → Shadow Replay → Same-seed Gate → Held-out seeds 1..20 → Promote 或回退。

**VLA 轨：**

- LIBERO-Pro + **Pi0.5**：`bash scripts/deployment/install_vla_env.sh --track libero-pro`
- RoboCasa + **GR00T**：需 `robocasa/`、`robosuite/`、`Isaac-GR00T/` 源码 checkout + ~10GB kitchen assets

**Rollout Runtime：**

```bash
python -m rollout_runtime.cli serve --config <preset> --host 127.0.0.1 --port 18730 --launch ray
```

## 交叉链接

- [paper-zetta](../../wiki/entities/paper-zetta.md)
- [paper-zeva](../../wiki/entities/paper-zeva.md)（同 org，不同部署期适应机制）
