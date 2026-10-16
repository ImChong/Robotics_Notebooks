# Gamma-Robot/bfm_ttc

- **URL：** <https://github.com/Gamma-Robot/bfm_ttc>
- **类型：** repo / research implementation
- **官方项目页：** <https://bfmttc.github.io/>
- **配套论文：** [Test-Time Compute for Humanoid Behavior Foundation Models（arXiv:2610.14896）](https://arxiv.org/abs/2610.14896)
- **默认分支：** `main`
- **语言：** Python
- **许可证：** 仓库主代码 Apache-2.0；SPIDER sampler 保留其上游 CC BY-NC 4.0，其他上游说明见 `THIRD_PARTY_NOTICES.md`。
- **核验日期：** 2026-10-16

## 一句话说明

论文官方实现，提供 Test-Time Command Optimization（TTCO）、采样优化器、BFM adapters、仿真运行入口、结果导出，以及 surrogate cost model 实验。

## 核心目录与入口

| 路径 | 作用 |
|------|------|
| `command_optimization` | 加载目标 motion、解码候选命令、仿真评估、receding-horizon 执行、计算指标与渲染视频 |
| `sampling_optimizers` | 提议候选命令，并依据标量 cost 更新采样分布 |
| `bfm_adapters` | 封装冻结的 BFM，构造 observation、推理策略、维护历史状态并映射动作到 PD command |
| `experiments/surrogate_cost/` | 拟合 surrogate cost model，并复现对应图表与实验 |
| `configs/experiments/` | 运行与批处理配置；README 提供 `bfm-ttc run` 和 `bfm-ttc render` CLI 示例 |

## 开放状态与复现边界

- **代码已公开且有可运行入口：** README 给出 Linux + CUDA GPU + Python 3.10+ 环境、`uv sync --frozen`、`bfm-ttc run` / `render` 示例。
- **不等于免配置权重包：** 运行示例前需按 README 准备 G1 robot assets 和 HEFT-PMG checkpoint；未发现项目页或仓库提供统一预训练 BFM checkpoint 下载页。
- **数据边界：** 仓库提供实验配置和示例动作，项目页无单独数据集发布链接；AMASS 评测动作须按其来源与许可另行准备。
- **运行边界：** README 的核心链路是仿真评估、重规划、指标与视频导出。论文/项目页真机片段为仿真优化命令的回放，不应将仓库描述为通用在线真机控制栈。
- **许可证边界：** 主仓库 Apache-2.0，SPIDER sampler 为上游 CC BY-NC 4.0；复用 sampler 前需遵守对应 notice。

## 运行路线（README）

1. 安装 Linux / Python / NVIDIA CUDA 环境，克隆仓库并执行 `uv sync --frozen`。
2. 准备 HEFT 仓库中的 G1 robot assets，并取得 HEFT-PMG checkpoint。
3. 用 `bfm-ttc run configs/experiments/single_motion.yaml --dry-run` 检查配置；随后运行同配置作 baseline / optimized 对照，再用 `bfm-ttc render` 导出视频。
4. 从 `report.json`、`paired_traces.npz`、轨迹导出和 `comparison.mp4` 查看 metrics 与结果。

## 关联资料

- [论文归档](../papers/ttco_test_time_compute_humanoid_bfm_arxiv_2610_14896.md)
- [项目页归档](../sites/ttco-test-time-compute-humanoid-bfm.md)
- [TTCO 统一论文实体](../../wiki/entities/paper-ttco-test-time-compute-humanoid-bfm.md)
