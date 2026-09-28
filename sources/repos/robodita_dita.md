# Dita（RoboDita/Dita）

> 来源归档（repo）

- **标题：** Dita — Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy
- **类型：** repo / vla / diffusion-transformer / oxe / open-source
- **链接：** <https://github.com/RoboDita/Dita>
- **许可证：** MIT
- **论文：** [arXiv:2503.19757](https://arxiv.org/abs/2503.19757) · 前序 [2410.15959](https://arxiv.org/abs/2410.15959) — [`sources/papers/dita_arxiv_2410_15959.md`](../papers/dita_arxiv_2410_15959.md)
- **项目页：** <https://robodita.github.io/> — [`sources/sites/robodita-github-io.md`](../sites/robodita-github-io.md)
- **入库日期：** 2026-09-28
- **一句话说明：** OXE 预训练、CALVIN/LIBERO/SimplerEnv 闭环评测、Franka **10-shot** `finetune_realdata.py`；checkpoint 在 Google Drive。
- **沉淀到 wiki：** [`wiki/entities/paper-dita-scaling-diffusion-transformer-vla.md`](../../wiki/entities/paper-dita-scaling-diffusion-transformer-vla.md)

---

## 开源状态（步骤 2.5）

| 项 | 状态（2026-09-28） |
|----|---------------------|
| 训练 | `scripts/train_diffusion_oxe.py`（OXE S3）；`train_diffusion_sim.py`（CALVIN/LIBERO/ManiSkill） |
| 微调 | `scripts/finetune_realdata.py`（`use_lora` True/False；10-shot pkl 列表） |
| 评测 | CALVIN 闭环 `+eval_only=1`；SimplerEnv 文档指向 [SimplerEnv](https://github.com/simpler-env/SimplerEnv) |
| 数据 | `ManiSkill2/` 子目录生成自定义 ManiSkill2 数据；LIBERO 需 OpenVLA 修改版数据集 |
| 环境 | Python 3.9、PyTorch 2、CUDA 12.1；`requirements_base.txt` / `requirements_calvin.txt` |
| 权重 | README「Model Checkpoints」Google Drive（含 Droid 预训练、无增广 ablation、Diffusion MLP Head 对照） |

**结论：** **已开源**；最短路径为下载 checkpoint + `finetune_realdata.py` 或 CALVIN/LIBERO 配置；OXE 从头预训练成本高。

## 仓库入口（对齐时序图）

| 路径 | 角色 |
|------|------|
| `scripts/train_diffusion_oxe.py` | OXE 多机 `torchrun` 预训练 |
| `scripts/finetune_realdata.py` | 真机 few-shot 微调 |
| `scripts/train_diffusion_sim.py` | CALVIN / LIBERO / ManiSkill 训练与 `eval_only` |
| `scripts/train_discrete_sim.py` | 离散动作对照（OC-VLA 相关） |
| `ManiSkill2/openx_utils/` | 自定义 OXE 风格 ManiSkill2 数据生成 |

## 关联资料

- [`sources/papers/dita_arxiv_2410_15959.md`](../papers/dita_arxiv_2410_15959.md)
- [`sources/sites/robodita-github-io.md`](../sites/robodita-github-io.md)
