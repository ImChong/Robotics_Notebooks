# GR-BC — 官方代码与失败轨迹再利用（2026-11-12 复核）

- **项目：** Graph-Reweighted Behavior Cloning with Human Interventions for Precise Robotic Manipulation / 带人工干预的图重加权行为克隆
- **论文：** [arXiv:2610.20504v1](https://arxiv.org/html/2610.20504v1)（2026-10-14，CoRL 2026）
- **项目主页：** https://gr-bc.github.io/
- **官方代码：** https://github.com/Gamma-Robot/GR-BC
- **代码许可：** Apache-2.0（上游权重按各自许可）
- **核对范围：** 2026-11-12 阅读官方 README、pyproject.toml 中的 CLI 入口，源文件 `src/lerobot/policies/grbc/{export,graph,progress,weights,training}.py`、`scripts/train_common.sh` 与 `docs/VALIDATION.md`；论文方法 §3–4、实验 §5、局限 §6 及附录 A–D 交叉核对。**未在本环境实际运行训练、导出或真机测试。**
- **通读：** 未完成
- **来源类型：** repo（官方公开实现的主要入口和权重计算链路核查，不代表通读整个 LeRobot 派生仓库）
- **对应 wiki：** [GR-BC](../../wiki/entities/paper-gr-bc.md)
- **相关原文：** [论文归档](../papers/gr_bc_arxiv_2610_20504.md)、[项目组织归档](../sites/gamma-robotics.md)

## 开源程度与可复现边界

**部分开源：** 已发布 graph weighting、权重导出、SmolVLA/X-VLA/π0.5 的加权 SFT 训练接口、进展图可视化与离线评测；官方论文使用的四任务数据、训练后策略权重、完整真机采集/部署脚本未随仓库发布。MCR ResNet50 checkpoint 需自行从原始发布方下载。README 仍把 RLT + GR-BC 在线 RL 列为未完成项。官方 `docs/VALIDATION.md` 报告 49 项测试通过、SmolVLA CPU 2 步和 RTX 5080 5 步冒烟训练；这**不是**论文真机成功率复现。

## 源码实读：失败片段怎么变成训练梯度

| 入口 / 模块 | 代码行为（已核查） | 用于诊断的边界 |
|---|---|---|
| `pyproject.toml` → `grbc-export` | CLI 实际指向 `lerobot.policies.grbc.export:main` | 不要把项目图误认为可直接端到端训练的单条命令 |
| `export.py:run` | `load_dataset → sample_nodes → extract_features → build_temporal_edges + build_multi_view_adaptive_similarity_edges → reverse_dijkstra → dense_progress → compute_graph_transition_weights` | 显式校验 `next.reward > 0`；若没有真实成功标签抛错 |
| `graph.py` | 三视角分别做 mutual-KNN，汇总候选；多视角满足 `min_camera_hits`、夹爪同态、局部尺度归一化距离阈值才连相似边；反向最短路从成功节点出发 | 跨轨迹相似边只是进展估计的虚拟桥接，**不是可执行动作**；多视角过滤未证明真实物理可达 |
| `progress.py:dense_progress` | 用有限剩余距离的分位数 `rmax` 归一化进展；`paper` 插值模式只在两端有限时插值 | 不可达段进展为 0，而不是凭空补出一条成功路径 |
| `weights.py:compute_graph_transition_weights` | `n_action_step` 视窗内同时评估终点净进展 `chunk_delta_progress` 和途中最大回退 `chunk_progress_dip`；回退超标或进展不足置零，合格片段用有界 tanh 权重 | 输出 `chunk_reject_reason`：`window_dip_below_start`、`delta_not_above_min_delta_progress` 或 `accepted_residual_gain`；尾部视窗裁剪到最后一帧 |
| `training.py:GRBCWeights` | 按数据集全局 `index` 读取 parquet，检查索引覆盖与 `episode_index/frame_index` 对齐，并以 `weighted_loss` 对逐样本损失加权 | 不是给“失败回合”统一正/负标签；权重直接作用在**实际执行过的动作块**上 |
| `scripts/train_common.sh` | `accelerate.commands.launch -m lerobot.scripts.lerobot_train`，通过 `--use_grbc` 与 `--grbc_weights_path` 注入权重 | 三种 backbone 配置不同；π0.5 需独立 Transformers 环境 |

## 官方 README 最短使用路径（示意，不构成实测）

```bash
export DATASET_ROOT=/absolute/path/to/lerobot-v3-task
export MCR_CKPT=/absolute/path/to/mcr_resnet50.pth
grbc-export --dataset "$DATASET_ROOT" --mcr-ckpt "$MCR_CKPT" --output-dir outputs/task
export WEIGHTS_PATH="$PWD/outputs/task/transition_weights.parquet"
export OUTPUT_DIR="$PWD/outputs/train_smolvla"
bash scripts/train_smolvla.sh
grbc-visualize --run outputs/task --episodes 0 1 6
grbc-evaluate --run outputs/task
```

数据必须包含三路同步相机、机器人状态/动作与真实成功标签；`grbc-export` 不篡改原始帧 `index`。需先建立兼容 CUDA/PyTorch 环境，依赖/模型检查点与真实数据均不随命令自动生成。X-VLA / π0.5 的完整硬件训练复现**未由官方验证文档证实**。

## 失败轨迹筛选的三个易误读点

1. **失败 episode 不等于全权重为零。** 若其中一段在目标可达图支持下稳定降低剩余代价，仍可成为高权重的正向模仿样本。
2. **人工接管不等于整段都应模仿。** 接管前的越界、修正途中的严重退步、接管后的有效恢复，统一经净进展与局部回退双阈值筛选。
3. **图内进展识别不等于在线泛化或安全保证。** 论文表 1 的评估轨迹参与构图；仅附录 B.3 有 Screw Tightening 的留出逐条插入诊断，图上短路径不编码力/碰撞风险。

## 核验链接

- [GR-BC 官方 README](https://github.com/Gamma-Robot/GR-BC/blob/main/README.md)
- [方法实现 `export.py`](https://github.com/Gamma-Robot/GR-BC/blob/main/src/lerobot/policies/grbc/export.py)
- [相似边与反向搜索 `graph.py`](https://github.com/Gamma-Robot/GR-BC/blob/main/src/lerobot/policies/grbc/graph.py)
- [稠密进展 `progress.py`](https://github.com/Gamma-Robot/GR-BC/blob/main/src/lerobot/policies/grbc/progress.py)
- [逐动作块权重 `weights.py`](https://github.com/Gamma-Robot/GR-BC/blob/main/src/lerobot/policies/grbc/weights.py)
- [加权损失 `training.py`](https://github.com/Gamma-Robot/GR-BC/blob/main/src/lerobot/policies/grbc/training.py)
- [训练入口 `train_common.sh`](https://github.com/Gamma-Robot/GR-BC/blob/main/scripts/train_common.sh)
- [官方验证范围](https://github.com/Gamma-Robot/GR-BC/blob/main/docs/VALIDATION.md)
