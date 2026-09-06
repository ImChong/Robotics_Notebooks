# SDPG — Self-Distilled Policy Gradient (verl)

> 来源归档（仓库 README 要点摘录）

- **标题：** SDPG: Self-Distilled Policy Gradient
- **类型：** repo
- **组织：** lauyikfung 等（UCLA / Princeton）
- **链接：** https://github.com/lauyikfung/SDPG
- **项目页：** https://lauyikfung.github.io/SDPG
- **论文：** arXiv:2606.04036
- **许可：** MIT
- **入库日期：** 2026-09-06
- **一句话说明：** 在 [verl](https://github.com/volcengine/verl) 上实现 GRPO / SDPG / OPSD / RLSD；privileged context 经 `[TEACHER_CONTEXT_TOKEN]` 注入；Qwen3-4B 数学 DAPO 复现脚本在 `examples/rpg2_trainer/`。
- **沉淀到 wiki：** [wiki/entities/paper-sdpg-self-distilled-policy-gradient.md](../../wiki/entities/paper-sdpg-self-distilled-policy-gradient.md)

---

## 依赖与运行面（README）

- 8× A100/H100/H200（脚本默认单节点 8 GPU）
- Ray：`ray start --head --num-gpus=8`
- 模型：`Qwen/Qwen3-4B`（或 `MODEL_PATH`）
- 数据：`math-dapo-teacher-shuffled-boxed.parquet`（SDPG）vs `math-dapo-noteacher-shuffled-boxed.parquet`（GRPO）
- 主脚本：`bash examples/rpg2_trainer/run_qwen3_4b_sdpg_boxed.sh`
- 超参：`BETA`、`ALPHA`、`KL_MODE`（默认 `urkl`）、`BETA_POSITIVE_ADV_ONLY`

---

## 方法对照（README 表）

| Method | Loss | Teacher | Ref model |
|--------|------|---------|-----------|
| GRPO | DAPO dual-clip PPO | None | Optional |
| **SDPG** | DAPO + full-vocab KL + α-reg | Current π(·\|c,x) | Yes (frozen) |
| OPSD | PPO-REINFORCE per-token | Frozen π_ref(·\|c,x) | Yes |
| RLSD | DAPO + evidence-reweighted adv | Current π(·\|c,x) | No |

---

## 开源状态

**已开源** — 训练脚本、数据格式与 verl 集成完整；属 **LLM 推理 RL** 栈，非机器人控制。

## 同名消歧

- 机器人视觉 RL：[HaoxiangYou/SDPG](https://github.com/HaoxiangYou/SDPG)（arXiv:2605.26478）

## 交叉链接

- [sources/papers/sdpg_self_distilled_policy_gradient_arxiv_2606_04036.md](../papers/sdpg_self_distilled_policy_gradient_arxiv_2606_04036.md)
- [sources/sites/sdpg-lauyikfung-website.md](../sites/sdpg-lauyikfung-website.md)
