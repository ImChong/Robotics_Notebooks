# raisimGym_nature（RaiLab KAIST · Nature RAIBO2）

- **URL：** <https://github.com/railabatkaist/raisimGym_nature>
- **类型：** 开源 RL 训练 / 奖励消融评测（Nature 论文配套）
- **维护方：** KAIST Robotics and Artificial Intelligence Laboratory（RaiLab）
- **收录日期：** 2026-09-25
- **Tags：** #quadruped #locomotion #reinforcement-learning #raisim #energy-efficiency #nature
- **论文：** <https://doi.org/10.1038/s41586-026-11102-5>
- **数据：** <https://doi.org/10.5281/zenodo.14825866>
- **许可证：** MIT

## 一句话

**RAIBO2** Nature 稿配套的 **RaiSim + raisimGymTorch** 训练与 **tester.py 奖励消融** 发布仓；与 sibling 目录 **`raisimLib`** 联编，**不含** RAIBO2 硬件 CAD/固件。

## 为什么值得保留

- 复现论文 **低耗散 locomotion policy** 与 **collision / Joule / 高度** 等奖励项 ablation 的 **官方入口**。
- 与 [`sources/repos/raisim.md`](raisim.md) 同一仿真栈，便于和 Hwangbo 组其它 raisimGym 项目对照。

## 核心内容（结构级）

| 模块 | 说明 |
|------|------|
| 布局 | `workspace/raisimGym_nature` + `workspace/raisimLib`（[raisimTech/raisimLib](https://github.com/raisimTech/raisimLib)） |
| Conda | `environment.yml` → 环境名 **`rsg_raibo_nature`**；PyTorch **2.5.1+cu121** |
| 训练 | `raisimGymTorchNature/` 下 env 与训练脚本（见 README） |
| 评测 | `raisimGymTorchNature/env/envs/rsg_raibo_nature/tester.py` — `--weight_type` 奖励消融、`--vis_target` 可视化 |
| 可视化 | **raisimUnity**（随 raisimLib 分发） |

## 开源边界（2026-09-25）

- **已发布：** 策略训练 + ablation 评测流程。
- **未发布：** RAIBO2 机械/电气 BOM、现场马拉松部署中间件；需 **raisimLib** 与 GPU/CUDA 环境。

## 相关引用

- [RAIBO2 Nature 实体页](../../wiki/entities/paper-raibo2-marathon-energy-efficient-quadruped.md)
- [论文摘录](../papers/raibo2_marathon_nature_s41586_026_11102_5.md)
- [Zenodo 数据集](../sites/zenodo_raibo2_marathon_dataset.md)
- [RaiSim 库归档](raisim.md)
