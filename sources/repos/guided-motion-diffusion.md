# Guided Motion Diffusion（korrawe/guided-motion-diffusion）

> 来源归档

- **类型：** repo
- **官方仓库：** <https://github.com/korrawe/guided-motion-diffusion>
- **项目页：** <https://korrawe.github.io/gmd-project/>
- **论文：** <https://arxiv.org/abs/2305.12577>
- **许可：** MIT（以仓库 `LICENSE` 为准；GitHub API 未识别 SPDX）
- **入库日期：** 2026-07-28
- **开源状态：** **已开源** — 训练、空间约束推理、评测代码与轨迹/动作预训练权重均已发布。

## 可运行入口（2026-07-28）

| 任务 | README 入口 |
|------|-------------|
| 生成/轨迹/关键帧/避障 | `python -m sample.generate ... --guidance_mode {kps,sdf,trajectory}` |
| 轨迹模型训练 | `python -m train.train_trajectory` |
| 动作模型训练 | `python -m train.train_gmd` |
| 文本到动作评测 | `python -m eval.eval_humanml --model_path ...` |
| 关键帧评测 | `python -m eval.eval_humanml_condition --model_path ...` |

仓库基线为 Ubuntu 20.04、Python 3.7 与 CUDA GPU；需下载 HumanML3D、SMPL、GloVe、评测器，并使用仓库提供的绝对根坐标预处理。

## 对 wiki 的映射

- [`paper-notebook-guided-motion-diffusion-for-controllable-human-m.md`](../../wiki/entities/paper-notebook-guided-motion-diffusion-for-controllable-human-m.md)
- [`gmd-project.md`](../sites/gmd-project.md)
- [`humanoid_pnb_guided-motion-diffusion-for-controllable-human-m.md`](../papers/humanoid_pnb_guided-motion-diffusion-for-controllable-human-m.md)
