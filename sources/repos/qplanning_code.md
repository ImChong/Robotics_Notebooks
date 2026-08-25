# varungiridhar/qplanning-code

> 来源归档

- **标题：** Q-Planning 官方实现
- **类型：** repo
- **代码：** <https://github.com/varungiridhar/qplanning-code>
- **项目页：** <https://varungiridhar.github.io/qplanning/>
- **论文：** <https://arxiv.org/abs/2608.21204>
- **入库日期：** 2026-08-25
- **一句话说明：** 冻结 BC 策略的 Q 加权 action 选择与 Q-only 在线自改进；`qplanning` CLI + LIBERO/RoboTwin YAML 配置；默认 BC 为 FastWAM。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Q-Planning](../../wiki/entities/paper-qplanning.md) | 实体归纳页 |
| [VLA](../../wiki/methods/vla.md) | 大 visuomotor BC 先验 + 价值引导推理 |
| [Action Chunking](../../wiki/methods/action-chunking.md) | chunk 级 Q 与重规划 |
| [LWD](../../wiki/methods/lwd.md) | 另一类「部署信号回灌」但更新整策略 |

## 复现入口（README 摘要）

```bash
git clone <repo> && cd qplanning_release
python -m venv .venv && source .venv/bin/activate
pip install -e ".[libero]"   # 或 ".[robotwin]"
cp .env.example .env
qplanning doctor --config configs/libero_10/qplanning_offline.yaml
qplanning eval --config configs/libero_10/qplanning_offline.yaml
qplanning self-improve --config configs/libero_10/self_improve.yaml
qplanning train-q --config configs/libero_10/train_q.yaml
```

- **评估：** `baseline.yaml`（纯 BC）vs `qplanning_offline.yaml`（Q 加权）；输出 `episodes.csv` + `summary.json`。
- **自改进：** 每轮 rollout → replay → **仅更新 Q**；可分 `--stage collect` / `finetune` 与 `--eval.shard`。
- **RoboTwin：** 需独立环境（SAPIEN、curobo），见 `docs/setup.md`。

## 开源状态

**已开源** — 训练、评测、自改进与报告脚本齐全；checkpoint 路径由 `.env` 指定。
