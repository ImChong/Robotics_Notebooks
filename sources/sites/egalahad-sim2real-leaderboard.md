# sim2real Motion Tracking Leaderboard

> 来源归档

- **类型：** benchmark / project site
- **链接：** https://egalahad.github.io/sim2real/leaderboard/
- **访问日期：** 2026-10-26
- **复核日期：** 2026-11-21
- **RoboParty 官方引用：** https://roboparty.com/documents/party-os?doc=mimiclite&lang=zh （MimicLite 使用指南 §6）
- **复现文档：** https://egalahad.github.io/sim2real/tutorials/cross-codebase-evaluation/
- **评测代码：** https://github.com/EGalahad/sim2real/tree/main/scripts/tracking_experiment
- **核对范围：** 完整阅读 Motion Tracking Leaderboard 页面、Cross-Codebase Tracking Evaluation 文档及 RoboParty 官方 MimicLite 指南第 6 章；逐段核对上游 README、run_motiondecode_eval.py、run_tracking_metrics_eval.py、compute_tracking_metrics.py 相关执行路径和 manifests/motiondecode.json，确认数据版本、失败条件、产物与指标定义。
- **通读：** 已完成
- **一句话说明：** 在共享 MuJoCo 与冻结动作集上对多个公开人形跟踪策略进行同口径对比的交互排行榜。

## 榜单范围

- 页面列出 15 个策略发布/配置条目，归并为 10 个项目家族；下表在新实体中逐行映射到一个唯一的项目详情节点，版本条目共用同一项目页，不重复创建实体。
- 默认测试集是 MotionDecode Locomotion-80、Manipulation-48、Ground-60 和 Dance-40；Legacy LAFAN-40、PHUMA-30、Root-90 保留作历史对照。
- 协议采用 seed 0、50 Hz policy frame、initial pause 0、无 runtime cap 和共同失败条件。每个 dataset 内按 motion 等权，页面 headline 对可用的 dataset means 等权平均。
- 指标包含局部 body position/orientation、Locomotion 的 Global Root、GPU training hours、wrist position/orientation、tracking return 与 progress。位置/姿态误差越低越好，return/progress 越高越好。
- Global Root 只对 Locomotion 报告；Manipulation 指 body/wrist tracking，不代表物体操作任务成功率。GPU hours 仅在有可辩护来源时填入；该榜单不是对所有训练成本的完整统计。
- 页面标注最近更新时间为 2026-09-15；复现前应在官方页确认榜单数据及协议是否有更新。

## 榜单快照（页面标注：2026-09-15）

以下为页面当前排序前三行；首列为榜单展示值，括号内分别是 Locomotion 与 Manipulation 数据集均值。误差越低越好，Tracking Return 与 Progress 越高越好。GPU Hours 仅抄录页面已填且带来源链接的值。

| 排名 / 策略 | Body Pos (mm) | Body Ori (rad) | Wrist Pos (mm) | Tracking Return | Progress (%) | GPU Hours |
|---|---:|---:|---:|---:|---:|---:|
| 1 · Mimic Lite v1.1 | 26.68 (25.11 / 28.25) | 0.088 (0.081 / 0.095) | 22.86 (22.15 / 23.57) | 1.852 (1.913 / 1.790) | 96.5 (99.3 / 93.6) | 50.1 |
| 2 · MimicLite-ROA | 31.98 (29.22 / 34.74) | 0.106 (0.095 / 0.118) | 33.21 (31.61 / 34.81) | 1.826 (1.886 / 1.766) | 96.6 (98.7 / 94.5) | 173.2 |
| 3 · MimicLite-PPO | 32.92 (31.52 / 34.32) | 0.109 (0.097 / 0.121) | 35.96 (33.25 / 38.66) | 1.836 (1.900 / 1.772) | 97.6 (100.0 / 95.2) | 92.3 |

页面没有为这三行提供 Global Root 的 Manipulation 值；横评时保留为缺失值，不作补零。

## 复现

官方教程固定 MotionDecode revision、ordered motion list 与文件 hashes；默认评测 Locomotion-80 / Manipulation-48，可选择四组全评。结果应随代码 commit、依赖锁、策略配置与原始轨迹保存。旧版 LAFAN/PHUMA/Root90 不是 MotionDecode 的替代集合。


## RoboParty 官方发布关系（2026-11-21 复核）

- [RoboParty Party OS 官方 MimicLite 指南 §6](https://roboparty.com/documents/party-os?doc=mimiclite&lang=zh) 直接链接这份动作跟踪榜单，并提供统一评测命令。它属于 RoboParty 使用的 **EGalahad/sim2real 跨代码库评测工具链**；并非另一套独立的 RoboParty leaderboard。
- [EGalahad/sim2real 官方 README](https://github.com/EGalahad/sim2real/blob/main/README.md) 在 Citation 中将维护团队署为 RoboParty Lab Team，榜单及其仓库可作为共同的一手入口。
- 榜单是 **Unitree G1 在统一 MuJoCo 中对参考动作的 Sim2Sim 跟踪结果**；不是跨机器人真机运控的综合排行榜，也不直接衡量接触操作任务成功率。

## 跑一次可复现的基准

在 [EGalahad/sim2real](https://github.com/EGalahad/sim2real) 仓库根目录执行（先准备 deploy YAML/ONNX 和 CPU inference 环境）：

```bash
uv sync --extra inference-cpu
uv run scripts/tracking_experiment/run_motiondecode_eval.py \
  --download \
  --policy my_policy=checkpoints/my_policy/policy.yaml \
  --output-dir outputs/my_policy_motiondecode \
  --max-workers 8
```

- 默认只跑 **Locomotion-80 + Manipulation-48**；要测全部四组，追加 `--splits locomotion manipulation ground dance`。以重复 `--policy name=path` 的方式横评多个策略。
- `--dry-run` 只执行下载/校验、打印命令，不运行仿真。缓存目录 `datasets/motiondecode`；`--skip-existing` 仅适合策略、仿真器、数据版本都不变的重复统计。
- 每组保存 `trajectories/`、`runs.csv`、`tracking_metrics.csv/json`、`summary.json`，根输出附 `motiondecode_summary.json`。复现实验必须保留 checkpoint、YAML、依赖锁及原始轨迹。
- 数据源是 ChingMu MotionDecode 的冻结 G1 120 Hz NPZ 小型评测集，不是全部原始 MotionDecode；源 revision 为 `80b489e0378b60bb44d495d5437475ce0e084283`，四组已转换数据的**各自 revision** 以 [manifests/motiondecode.json](https://github.com/EGalahad/sim2real/blob/main/scripts/tracking_experiment/manifests/motiondecode.json) 为准。运行器读取 `checksums.json` 对文件算 SHA256，并检查 `motion-list.txt` 的数量与唯一性。原始数据访问条款仍适用，勿擅自改采样或迁移为其他机器人数据。

## 真实的评分边界与实现依据

- 固定随机种子 0、策略帧率 50 Hz、无初始暂停、无单条运行时长上限。**失败阈值**为骨盆高度差严格大于 **0.3 m**，或骨盆投影重力 Z 误差严格大于 **0.8**，与训练时的 episode termination 无关；root 位置误差本身不触发失败。
- `Progress` 与 `Tracking Return` 同在首个失败帧截断，return 不计失败帧；局部位置/姿态误差只平均有效帧。不能只追求更低 Body Pos 而忽略失败早退。Tracking Return 以 BeyondMimic 风格的位置/姿态指数奖励（尺度 0.3/0.4）对完整参考时长归一化，值域 0–2。
- `Body Pos / Wrist Pos` 是相对当前骨盆 XY/yaw 对齐（**保留绝对 Z**）后的局部误差；`Global Root` 是 **Locomotion** 专属的固定起始 XY/yaw 对齐后**同一有效时间终点 XYZ 距离**。它不是整段 root 轨迹均值；旧版其他 root 约定必须以原始轨迹重算，不可直接比较。
- 官方 `scripts/tracking_experiment/README.md` 的部分历史描述仍有 **0.25 m** 阈值、轨迹平均 root 的旧口径；本次以 [当前评分脚本](https://github.com/EGalahad/sim2real/blob/main/scripts/tracking_experiment/compute_tracking_metrics.py) 中 `TERMINATION_PELVIS_HEIGHT_THRESHOLD_M = 0.3`、以及[新评测文档](https://egalahad.github.io/sim2real/tutorials/cross-codebase-evaluation/)的 endpoint 定义为准。
- [运行器](https://github.com/EGalahad/sim2real/blob/main/scripts/tracking_experiment/run_motiondecode_eval.py) → [批量评测](https://github.com/EGalahad/sim2real/blob/main/scripts/tracking_experiment/run_tracking_metrics_eval.py) → `sim2real.sim_env.integrated_sim2sim` → [离线评分](https://github.com/EGalahad/sim2real/blob/main/scripts/tracking_experiment/compute_tracking_metrics.py)。这些是**源码核验的执行顺序**，不是独立重跑所得数据。
