# Beyond the Remembered World: Predictive 4D Belief for Persistent Navigation in Evolving Worlds（arXiv:2609.39166）

> 来源归档（原文通读 · 支撑 `wiki/entities/paper-evolvingnav.md`）

- **标题：** Beyond the Remembered World: Predictive 4D Belief for Persistent Navigation in Evolving Worlds
- **项目名：** EvolvingNav / P4D-Nav / EvoWorld-Bench
- **作者：** Mingjian Gao, Zhaocheng Li, Haoyang Huang, Wenqiao Zhang, Yingjie Niu, Hao Zhou, Chao Li, Juncheng Li, Siliang Tang, Yueting Zhuang（Deeprobotics / 浙江大学 / UCSD / CUHK）
- **arXiv：** <https://arxiv.org/abs/2609.39166>（2026-10 v4，2026-11 通过 arXiv PDF 获取全文）
- **项目页：** <https://developer.deeprobotics.cn/research/articles/evolving-nav/index.html> · <https://zju4embodiedai.github.io/EvolvingNav/>
- **代码：** <https://github.com/ZJU4EmbodiedAI/EvolvingNav>（`/code` 目录开源可运行实现）
- **交互演示：** <https://huggingface.co/spaces/ZJU4EmbodiedAI/EvolvingNav>
- **一句话说明：** 针对「过去记得的目标在到达前可能已被移动或遮挡」的演化环境持久化导航，提出预测式 4D 持久-迁移信念（P4D-Nav）与事件驱动闭合更新回路，并构建基于真实人类作息先验的 54 场景 × 90 天 EvoWorld-Bench 基准（803.68k 可执行任务）。
- **核对范围：** 通读 arXiv PDF v4 全文 §1–§6（问题形式化、P4D-Nav 因果 4D 记忆/CTTransformer/转移核与 ETA 到达预测/可见性条件更新、EvoWorld-Bench 构造协议、N1–N5 与 VLN/EQA 实验及消融、LYNX M20/X30/Lite3 真机验证）及 `ZJU4EmbodiedAI/EvolvingNav/code` 仓库入口。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **核心设定（§1–§2）：** 静态地图与“记住即真”的 3D 场景图假设在长期部署中失效；物品会随时间在候选位置间迁移（relocation）或停留在上次观测处（persistence），且搜索成本取决于智能体**到达目标候选点时刻（ETA）**的占用概率而非查询起点时刻概率。
- **P4D-Nav 四模块机制（§3）：**
  1. **因果 4D 记忆（Causal 4D Memory）：** 将共享坐标系下的 RGB-D 观测与事件增量物化为带版本戳与时间戳的实体历史，避免覆盖或丢弃候选位置演变证据。
  2. **连续时间历史编码与持久–迁移分解：** 使用 CTTransformer 编码最近 K 个因果事件 token（含 `seen`/`notseen` 证据、时间增量与可见性覆盖），输出持久概率 `ρ_q`（停留在末次观测点）与迁移分布 `π_q`（各已知候选点 + `unknown` 桶）构成的基线信念 `b⁻`。
  3. **学习转移核与到达时刻预测（`b_arr`）：** 通过行归一化转移头 `K_θ(Δt)` 随流逝时间演化分布，并前瞻预计行程时间 `τ_ETA` 预测到达时刻占用概率 `b_arr = b^+ · K_θ(τ_ETA)`，指导冻结 VLM 控制器权衡“到达探测概率 / 行程代价 / 检查次数”。
  4. **可见性限定测量更新与候选重开：** 短段移动采集 RGB-D 仅当新增可见性覆盖超阈值时才执行贝叶斯更新（未检出时使用标定过的检测召回率 `r̂` 构造似然 `1 - r̂`，`unknown` 似然为 1），并支持后续新线索重新激活曾被否决的候选点，不做硬删除。
- **EvoWorld-Bench（§4）：** 基于 **54 个 HSSD 室内场景** 模拟 **90 天**演化，由真实人类行为数据集（CASAS、ARAS、OPPORTUNITY、HD-EPIC、ParaHome、HOMER+）标定作息与物品迁移规律，共 **803,680 个可执行任务**，覆盖 **P（预测）、N1–N5（跨日/单日静态检索、演化检索、在线动态迁移 N4、多跳协同等）、VLN 与 EQA**。
- **主要评测与真机结果（§5）：**
  - **EvoWorld N1–N5：** 首次检查成功率（First-Inspection SR）**61.32%**，预算搜索 SR **86.18%**；其中目标在执行过程中动态迁徙的 **N4 Online Dynamic** 子集 SR 达 **65.2%**。
  - **真机验证：** 在云深处 **LYNX M20** 开展 **64 次物理对比测试**（预算 Search SR **48.4%** vs Retrieval+Reasoning 基线 **37.5%**），并展示 **X30** 与 **Lite3** 上的室内巡逻与文件递送测试。
- **源码入口（`ZJU4EmbodiedAI/EvolvingNav/code`）：** 基于 Python 3.11 + Habitat-Sim 0.3.3，采用 Grounding DINO + SAM2 感知与 GPT-5.6-Luna / `--controller utility` 控制器；关键入口覆盖 `scripts/pack_p4d_hssd_records.py`、`scripts/train_p4d_belief.py`、`scripts/train_transition.py`、`scripts/collect_calibration.py`、`scripts/fit_calibration.py`、`python -m evolvingnav.run` 与 `python -m evolvingnav.verify_visual`。

## 对 wiki 的映射

- [paper-evolvingnav](../../wiki/entities/paper-evolvingnav.md)
- [deeprobotics-evolvingnav site archive](../sites/deeprobotics-evolvingnav.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2609.39166>
- 仓库：<https://github.com/ZJU4EmbodiedAI/EvolvingNav>
