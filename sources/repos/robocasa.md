# robocasa

> 来源归档

- **标题：** RoboCasa / RoboCasa365
- **类型：** repo
- **来源：** UT Austin（robocasa 组织）
- **链接：** https://github.com/robocasa/robocasa
- **主页：** https://robocasa.ai/
- **文档：** https://robocasa.ai/docs/introduction/overview.html
- **Leaderboard：** https://robocasa.ai/leaderboard.html
- **Stars：** ~1.7k+（2026-09）
- **入库日期：** 2026-09-06
- **一句话说明：** 大规模厨房日常任务仿真框架（MuJoCo / robosuite 后端）；RoboCasa365 提供 365 任务、2500+ 场景、2200+ 小时演示与公开 leaderboard。
- **代码：** https://github.com/robocasa/robocasa（**已开源**，代码 MIT；资产与数据集 CC BY 4.0）
- **沉淀到 wiki：** 是 → [`wiki/entities/robocasa.md`](../../wiki/entities/robocasa.md)

---

## 核心定位

- **后端：** [robosuite](https://github.com/ARISE-Initiative/robosuite)（**须用 master 分支**）
- **场景：** 以厨房为中心的人类居家环境；RoboCasa365 扩至 **2500+** 独特厨房布局
- **任务：** LLM（GPT-4）辅助定义 **365** 日常复合任务；**65** 原子技能任务覆盖 10 项基础能力
- **资产：** 3200+ 物体（Objaverse、Lightwheel AI、Luma AI 生成等）；AI 纹理域随机化（MidJourney）
- **数据：** **600+ h** 人类遥操作演示 + **1600+ h** 自动轨迹合成机器人数据（合计 **2200+ h**）
- **评测：** [RoboCasa365 Leaderboard](https://robocasa.ai/leaderboard.html)（50 任务多任务学习榜，截至 2026-09-01）

---

## 版本里程碑（README）

| 版本 | 日期 | 要点 |
|------|------|------|
| **v1.0.1** | 2026-05-12 | 全任务 horizon **1.5×**；评测须对齐此版 |
| **v1.0** | 2026-02-18 | RoboCasa365：365 任务 / 2500 场景 / 2200+ h 数据 / 公开榜 |
| **v0.2** | 2024-10-31 | robosuite v1.5 后端；自定义机器人组合、复合控制器、逼真渲染 |

2026-07-07：复合任务数据集增加 **逐帧 subtask 标注**（subtask index、原子技能名、stage、自然语言指令）。

---

## 安装要点（README）

```bash
conda create -c conda-forge -n robocasa python=3.11
conda activate robocasa
git clone https://github.com/ARISE-Initiative/robosuite && cd robosuite && pip install -e .
git clone https://github.com/robocasa/robocasa && cd robocasa && pip install -e .
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets   # ~10GB
```

---

## Gym 入口示例

```python
import gymnasium as gym
import robocasa

env = gym.make(
    "robocasa/PickPlaceCounterToCabinet",
    split="pretrain",  # 或 "target"
    seed=0,
)
```

演示脚本：`robocasa.demos.demo_tasks` / `demo_kitchen_scenes` / `demo_objects` / `demo_teleop`。

---

## Leaderboard 快照（2026-09-01）

50 任务多任务榜 Overall（Human300 预训练 → 50 target 任务评测）：

| Rank | Policy | Overall | Atomic-Seen | Composite-Seen | Composite-Unseen |
|------|--------|---------|-------------|----------------|------------------|
| 1 | Xiaomi-Robotics-1 | 57.4 | 80.2% | 57.1% | 32.1% |
| 2 | ABot-M0.6 | 46.6 | 79.4% | 48.3% | 7.9% |
| 5 | RLDX-1 | 36.0 | 67.6% | 27.9% | 8.5% |
| 7 | GR00T N1.5 | 23.9 | 50.7% | 14.8% | 2.7% |
| 10 | π0.5 | 16.9 | 39.6% | 7.1% | 1.2% |
| 13 | Diffusion Policy | 6.1 | 15.7% | 0.2% | 1.3% |

Splits：**Atomic-Seen 18** / **Composite-Seen 16** / **Composite-Unseen 16**（后者预训练未见过，测零样本泛化）。

---

## 与生态关系

- **Lightwheel LW-BenchHub：** 138+ RoboCasa/LIBERO 厨房任务经 Isaac Lab-Arena EnvHub 发布（见 [lw_benchhub](../repos/lw-benchhub.md)）
- **Isaac Lab-Arena：** NVIDIA 博客以 RoboCasa 类任务对比 GPU 并行 vs MuJoCo 顺序评测加速
- **DexBench：** 工业真机灵巧规格（RLWRLD × NVIDIA），与 RoboCasa 厨房仿真 **不同赛道**，勿混比 SR

---

## 开源核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| GitHub 仓 | **已开源**（MIT + CC BY 4.0 资产） |
| 可运行 | `pip install -e` + ~10GB 厨房资产下载 |
| Leaderboard | 公开提交审核制；训练配置透明但跨架构不可直接比 |
| 论文 | RSS 2024（原版）+ ICLR 2026（RoboCasa365） |

---

## 对 wiki 的映射

- [RoboCasa](../../wiki/entities/robocasa.md)
- [DexBench](../../wiki/entities/dexbench.md) — 工业真机规格对照
- [Isaac Lab-Arena](../../wiki/entities/isaac-lab-arena.md)
- [LW BENCHHUB TOUR](../../wiki/entities/lw-benchhub-tour.md)
