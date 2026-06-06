# ihcr/learning_to_adapt

> 来源归档

- **标题：** Learning to Adapt through Bio-Inspired Gait Strategies for Versatile Quadruped Locomotion（官方代码与图表数据）
- **类型：** repo
- **组织：** IHCR（Imperial Humanoid and Cognitive Robotics；论文作者 Joseph Humphreys, Chengxu Zhou）
- **代码：** <https://github.com/ihcr/learning_to_adapt>
- **论文：** <https://doi.org/10.1038/s42256-025-01065-z>（摘录见 [`sources/papers/learning_to_adapt_nature_2025.md`](../papers/learning_to_adapt_nature_2025.md)）
- **核心依赖：** 子模块 [`ihcr/bio_gait`](https://github.com/ihcr/bio_gait)（Colcon 编译）；**RaiSim** ≥1.1.6；PyTorch（测试 1.10.2+cu113）；Ubuntu 20.04 + Python 3.9
- **入库日期：** 2026-06-06
- **一句话说明：** Nature MI 2025 四足 **多步态 DRL + 生物力学步态选择** 的 **demo 脚本、配置与论文图表复现数据**；非 IsaacGym 栈，适合对照 **RaiSim + 分层 πG/BGS/πL** 实现。

## 仓库内容

| 路径 / 命令 | 说明 |
|-------------|------|
| `scripts/run_demo.py` | 加载 `configs/loco_bio_gs_unified.yaml` 运行可视化 demo |
| `data/` | 论文各 figure 对应原始数据（标签与正文一致） |
| `src/bio_gait`（子模块） | BGS 与 gait 参考生成核心实现 |

### 可用 demo

| 名称 | 场景 |
|------|------|
| `sprint` / `sprint_terr` | 直线冲刺；平地 vs rough 上 **效率 + 稳定性** 步态选择 |
| `stresstest` / `stresstest_terr` | 复杂速度命令；rough 下 locomotion + gait selection 联合适应 |
| `planks` | 松散木材；**辅助步态（pronk/bound/hop/limp）** 与名义步态联合恢复 |
| `allgaits` / `allgaits_terr` | 手动循环 **全部 8 步态** |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Learning to Adapt 论文实体](../../wiki/entities/paper-learning-to-adapt-bio-inspired-quadruped-gait.md) | 方法正文与实验归纳 |
| [Gait Generation](../../wiki/concepts/gait-generation.md) | **BGS + πG** 作为 RL 步态调度与切换的 Nature 级参考 |
| [Locomotion 任务页](../../wiki/tasks/locomotion.md) | **盲本体多步态 sim2real** 代码侧入口 |
| [Walk These Ways（MoB）](../../wiki/entities/paper-walk-these-ways-quadruped-mob.md) | 同为 **多样步态/行为** 但接口为 **行为参数 b** 而非 **Γ\* + BGS** |

## 对 wiki 的映射

- 与论文源文件成对维护：**[`sources/papers/learning_to_adapt_nature_2025.md`](../papers/learning_to_adapt_nature_2025.md)** → **[`wiki/entities/paper-learning-to-adapt-bio-inspired-quadruped-gait.md`](../../wiki/entities/paper-learning-to-adapt-bio-inspired-quadruped-gait.md)**
