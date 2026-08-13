# FACT（Bariona/FACT）

> 来源归档（repo）

- **标题：** FACT — Failure-Aware Causal Training for World-Action Models
- **类型：** repo / world-action-models / joint-wam / robotwin
- **来源：** Bariona（UCSD 作者仓）
- **链接：** <https://github.com/Bariona/FACT>
- **论文：** [arXiv:2608.10232](https://arxiv.org/abs/2608.10232) — 归档见 [`sources/papers/fact_arxiv_2608_10232.md`](../papers/fact_arxiv_2608_10232.md)
- **项目页：** <https://fact-wam.github.io/> — [`sources/sites/fact-wam-github-io.md`](../sites/fact-wam-github-io.md)
- **权重：** <https://huggingface.co/Bariona/fact-wam>
- **许可证：** Apache-2.0
- **入库日期：** 2026-08-13
- **一句话说明：** FACT 官方端到端 RoboTwin 管线：数据准备 → 训练 → inference server → 闭环评测；基于 Wan2.2-TI2V-5B。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-fact.md`](../../wiki/entities/paper-fact.md)

---

## 开源状态（步骤 2.5）

| 项 | 状态（2026-08-13） |
|----|-------------------|
| 训练 / 推理代码 | **已开源**（`scripts/train`、`scripts/inference_server`） |
| 评测 | `evaluation/robotwin/` + launch server/client |
| 权重 | HF `Bariona/fact-wam`（transformer + `norm_stats_delta.json`） |
| 数据 | HF dataset `Bariona/robotwin-v2` |
| 依赖 | Wan2.2 Diffusers、RoboTwin 仿真、conda `setup_env.sh` |
| 许可证 | **Apache-2.0** |

**结论：** **已开源可运行实现**；完整复现需 Wan 基座 + RoboTwin 环境与数据。

---

## README 宣称的技术栈 / 入口

| 组件 | 路径 / 命令 |
|------|-------------|
| 环境 | `bash setup_env.sh` → conda env `fact` |
| 数据准备 | `python -m scripts.prepare_robotwin` |
| VAE latent 缓存 | `python -m scripts.compute_vae_latents` |
| 训练 | `python -m scripts.train --config world_action_model.configs.robotwin.config` |
| 推理服务 | `python -m scripts.inference_server`（可 `--skip_future_state_value` 仅动作） |
| 闭环评测 | `evaluation/robotwin/launch_server.sh` + `launch_client.sh` / `eval_all_tasks.sh` |
| 模型包 | `world_action_model/`（configs、trainer、transforms） |

## 关联资料

- 论文归档：[`sources/papers/fact_arxiv_2608_10232.md`](../papers/fact_arxiv_2608_10232.md)
- 项目页：[`sources/sites/fact-wam-github-io.md`](../sites/fact-wam-github-io.md)
- Wiki 实体：[wiki/entities/paper-fact.md](../../wiki/entities/paper-fact.md)
