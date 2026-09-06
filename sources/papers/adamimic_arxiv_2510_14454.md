# Towards Adaptable Humanoid Control via Adaptive Motion Tracking（AdaMimic，arXiv:2510.14454）

> 来源归档（ingest）

- **标题：** Towards Adaptable Humanoid Control via Adaptive Motion Tracking
- **缩写 / 框架：** **AdaMimic**（Adaptive Motion Tracking）
- **类型：** paper / humanoid / motion-tracking / sim2real / keyframing / time-warping
- **arXiv：** <https://arxiv.org/abs/2510.14454>（PDF：<https://arxiv.org/pdf/2510.14454>）
- **会议：** IEEE ICRA 2026（**Oral**）
- **项目页：** <https://taohuang13.github.io/adamimic.github.io/>
- **代码：** <https://github.com/InternRobotics/AdaMimic>（CC BY-NC-SA 4.0）— 归档见 [`sources/repos/adamimic.md`](../repos/adamimic.md)
- **作者：** Tao Huang、Huayi Wang、Junli Ren、Kangning Yin、Zirui Wang、Xiao Chen、Feiyu Jia、Wentao Zhang、Junfeng Long、Jingbo Wang†、Jiangmiao Pang†
- **机构：** 上海人工智能实验室（Shanghai AI Lab）；上海交通大学（SJTU）
- **入库日期：** 2026-09-06
- **一句话说明：** 从**单条**参考运动出发，经关键帧稀疏化/轻量编辑构造增强集，两阶段 RL（固定相位跟踪 + phase/tracking 双适配器 time warping）在 Unitree G1 上实现可适应的敏捷全身模仿；Isaac Gym + PPO 训练，真机 FastLIO 定位部署。

## 开源状态（步骤 2.5）

- **项目页核查（2026-09-06）：** 页眉链出 GitHub [InternRobotics/AdaMimic](https://github.com/InternRobotics/AdaMimic)；无独立 Hugging Face / 权重托管页。
- **仓库核查：** README 含 `conda env create`、`legged_gym/scripts/train.py`（stage1/stage2）与 `play.py`；依赖 Isaac Gym、`rsl_rl`、`legged_gym`；`g1_dof27` 任务配置齐全。
- **结论：** **已开源**（训练 / 推理脚本与基线配置完整）；许可 **CC BY-NC-SA 4.0**，**禁止商业使用**；预训练 checkpoint 是否随仓发布以 README 为准。

## 摘录 1：问题与主张（§I）

- **痛点：** motion prior（AMP 等）适应性强但模仿精度差；motion tracking（DeepMimic 等）精度高但依赖大规模训练动作与测试时目标轨迹。
- **主张：** **单条参考运动** + 关键帧稀疏化与轻量全局编辑 → Stage I 稀疏全局 + 稠密局部双 critic 跟踪 → Stage II **phase adapter**（调制 \(\Delta\phi\)）与 **tracking adapter**（补偿低层动作）实现 time warping。
- **平台：** Unitree G1 **29 DoF**；真机锁定 **腰部 roll/pitch**；策略 50 Hz，PD 500 Hz，FastLIO 里程计 10 Hz。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-adamimic.md`](../../wiki/entities/paper-adamimic.md)；与 [BeyondMimic](../../wiki/methods/beyondmimic.md)、[AdaPT](../../wiki/entities/paper-adapt.md)、[YAHMP](../../wiki/entities/paper-yahmp.md) 互链。

## 摘录 2：方法要点（§IV）

| 模块 | 要点 |
|------|------|
| **数据** | 人体视频 → GVHMR → SMPL 重定向；单 clip 抽 \(N\) 个语义关键帧，子集 \(\Phi^{\mathrm{edit}}\) 做全局编辑（局部关节路径不变） |
| **Stage I** | 固定 \(\Delta\phi\)；稀疏全局奖励仅在 \(\phi\in\Phi^{\mathrm{key}}\) 激活 + 稠密局部奖励；**双 value critic** 分别估计稀疏/稠密回报 |
| **Stage II** | \(\Delta\phi_k^{\mathrm{ada}}=\Delta\phi_k+\Delta\phi_k^{\Delta}\)；跟踪动作残差 \(\Delta a\) 经 \(\Delta\phi^{\Delta}\) 缩放退化到 Stage I 动作；冻结 Stage I 权重 |
| **训练** | Isaac Gym，4096 env，PPO，3 层 MLP；L2C2 正则；\(\bm{w}=(1,0.5)\) 稀疏/稠密权重 |
| **任务** | 7 类：远跳、跳高、三级跳、台阶上下跳、网球击球、羽毛球击球等；仿真 easy/hard 适应区间见 Table I |

**对 wiki 的映射：** 实体页含流程总览 Mermaid + 源码运行时序图（train stage1 → stage2 → play）。

## 摘录 3：实验与真机（§V / Table III–IV）

| 轴 | AdaMimic 要点（仿真 overall） |
|----|------------------------------|
| vs AMP-Style | 成功率 **86.8%** vs 82.7%；局部/全局误差显著更低 |
| vs DeepMimic-Adapt | hard 适应成功率 **74.2%** vs 74.8%（接近），但全局稀疏误差 **99.8** vs **142.8** mm |
| vs AdaMimic-Stage1 | Stage II 适配器将 easy 成功率 **96.7%→99.6%**，局部误差 **43.4→30.3** mm |
| 真机（Table IV） | hard 适应：远跳 **5/6**、跳高 **5/6**（DeepMimic-Adapt 跳高 **0/6**）；网球/羽毛球 hard **6/6** |

**对 wiki 的映射：** 结论节强调「单 clip + 关键帧 + time warping」相对规则编辑与固定相位跟踪的 hard 适应增益。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-adamimic.md`**（完整实体页）。
- 新建 **`sources/sites/adamimic-github-io.md`**、**`sources/repos/adamimic.md`**。
- 更新 **`paper-notebook-adamimic`** 占位页链向完整实体；合并重复题名占位页。
- 交叉更新 [paper-adapt](../../wiki/entities/paper-adapt.md)（发球残差跟踪谱系）与分类索引。
