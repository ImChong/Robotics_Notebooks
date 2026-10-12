# Human2Dex 官方仓库

> 来源归档

- **标题：** zgates2/Human2Dex_
- **类型：** repo
- **代码：** <https://github.com/zgates2/Human2Dex_>
- **论文：** <https://arxiv.org/abs/2610.13119>（[论文归档](../papers/human2dex_arxiv_2610_13119.md)）
- **项目页：** <https://human2dex.pages.dev/>（[站点归档](../sites/human2dex-pages-dev.md)）
- **许可：** MIT（LICENSE 文件版权行为 Columbia AI and Robotics Lab 2023，疑似继承自上游 UMI/Diffusion Policy 代码）
- **入库/核查日期：** 2026-10-12（浅克隆；最近提交 2026-09-29；约 1,000 个文件）
- **沉淀到 wiki：** [`wiki/entities/paper-human2dex-wrist-centric-dexterous.md`](../../wiki/entities/paper-human2dex-wrist-centric-dexterous.md)

## 仓库定位

README 称它合并两个内部代码库：DexUMI 的数据采集与处理，以及 Dex_Data-Scaling-Laws-Infer 的训练与真机推理。

## 入口

| 阶段 | 路径 |
|------|------|
| 采集 | `teleop/teleop_mujoco.py`、`teleop/teleop_real.py`、`teleop/collect_data.py`、`teleop/collect_data_fused.py` |
| 处理 | `glove_aug_pipeline/run_human2dex_pipeline.sh`、`tools/convert_pkl_to_training_zarr.py`（输出 `dataset.zarr.zip`） |
| 训练 | `train_scripts/*.sh` → `train.py` → `diffusion_policy/` |
| 推理 | `eval_real_franka_pts21.py`、`eval_real_franka_o6.py`、`eval_real_dexumi.py`；`real_inference_*.py`；`scripts_real/`（标定、控制） |
| 硬件 | `o6_right_hand/`、`linker_o6/`、`wuji_retargeting/`、`dex-retargeting/` |

`train_scripts/` 下有按任务命名的脚本（如 `glove_cup_in_cup_green_red_cup_pocket_20hz.sh`、`glove_egg_carton_no_object_20hz.sh`）。

## 边界

- 数据、日志、checkpoint、生成的 Zarr、SAM3 权重被 .gitignore 排除；配置里保留机器特定数据集与标定路径，需自行修改。
- 未逐脚本运行验证；未发现公开的训练好权重。
