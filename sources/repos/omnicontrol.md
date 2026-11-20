# OmniControl（neu-3dvi/OmniControl，原 neu-vi/omnicontrol）

> 来源归档

- **类型：** repo
- **官方仓库：** <https://github.com/neu-3dvi/OmniControl>（原 `https://github.com/neu-vi/omnicontrol` 重定向）
- **项目页：** <https://neu-vi.github.io/omnicontrol/>（2026-11-20 返回 HTTP 404）
- **论文：** <https://arxiv.org/abs/2310.08580>
- **许可：** MIT
- **入库日期：** 2026-07-28（2026-11-20 复核）
- **通读：** 已完成
- **核对范围：** 2026-11-20 通读 `neu-3dvi/OmniControl`（原 `neu-vi/omnicontrol`）`README.md`（L1–150+）及推理入口 `sample/generate.py`（L1–200），核对 `collate_all` / `hint` 构造、`load_model_wo_clip`、`ClassifierFreeSampleModel`、`diffusion.p_sample_loop` 与 `recover_from_ric -> model.rot2xyz -> simple_eval` 评估链路。
- **开源状态：** **已开源** — HumanML3D 训练/推理/评测与 checkpoint 已发布；KIT-ML checkpoint 和交叉关节组合评测尚未发布。

## 可运行入口（2026-07-28）

| 任务 | README 入口 |
|------|-------------|
| 条件推理 | `python -m sample.generate --model_path ./save/omnicontrol_ckpt/model_humanml3d.pt` |
| 训练 | `python -m train.train_mdm ... --resume_checkpoint ...` |
| 全设置评测 | `./eval_omnicontrol_all.sh ...` |
| 单关节/密度评测 | `./eval_omnicontrol.sh ... <joint-id> <density>` |
| 空间引导 | `diffusion/gaussian_diffusion.py` |
| 真实感引导 | `model/cmdm.py` |

环境要求 Python 3.7、CUDA GPU、HumanML3D/KIT-ML 与 SMPL/GloVe/评测器资产；README 报告全设置单 GPU 评测约 45 小时。

## 对 wiki 的映射

- [`paper-notebook-omnicontrol-control-any-joint-at-any-time-for-hu.md`](../../wiki/entities/paper-notebook-omnicontrol-control-any-joint-at-any-time-for-hu.md)
- [`omnicontrol-project.md`](../sites/omnicontrol-project.md)
- [`humanoid_pnb_omnicontrol-control-any-joint-at-any-time-for-hu.md`](../papers/humanoid_pnb_omnicontrol-control-any-joint-at-any-time-for-hu.md)
