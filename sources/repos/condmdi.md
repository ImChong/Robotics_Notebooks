# CondMDI（setarehc/diffusion-motion-inbetweening）

> 来源归档

- **类型：** repo
- **官方仓库：** <https://github.com/setarehc/diffusion-motion-inbetweening>
- **项目页：** <https://setarehc.github.io/CondMDI/>
- **论文：** <https://arxiv.org/abs/2405.11126>
- **许可：** MIT（以仓库 `LICENSE` 为准；GitHub API 未识别 SPDX）
- **入库日期：** 2026-07-28
- **开源状态：** **已开源** — 训练、条件生成、评测代码及 HumanML3D 预训练权重均可获取。

## 可运行入口（2026-07-28）

| 任务 | README 入口 |
|------|-------------|
| 条件推理 | `python -m sample.conditional_synthesis --model_path ... --edit_mode ...` |
| 推理期插补/引导 | `python -m sample.edit ... --imputate --reconstruction_guidance` |
| 条件训练 | `python -m train.train_condmdi --keyframe_conditioned` |
| 评测 | `python -m eval.eval_humanml_condmdi --model_path ...` |
| SMPL 网格导出 | `python -m visualize.render_mesh --input_path ...` |

环境基线为 Ubuntu 20.04、Python 3.7、CUDA 11.7、PyTorch 1.13.1；数据需按 GMD 的绝对根表示重建 HumanML3D。

## 对 wiki 的映射

- [`paper-notebook-flexible-motion-in-betweening-with-diffusion-mod.md`](../../wiki/entities/paper-notebook-flexible-motion-in-betweening-with-diffusion-mod.md)
- [`condmdi-project.md`](../sites/condmdi-project.md)
- [`humanoid_pnb_flexible-motion-in-betweening-with-diffusion-mod.md`](../papers/humanoid_pnb_flexible-motion-in-betweening-with-diffusion-mod.md)
