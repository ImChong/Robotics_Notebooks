# EmbodiedSmith: Scaling Embodied Data through Recursive Self-Improvement Flywheel in Simulation

- **作者：** Yikai Qin, Yifei Deng, Mingjian Liang, Wenxuan Song, Zepeng Lin, Zhiyi Jiang, Jiajun Fu, Qiao Sun, Huashuo Lei, Xicheng Gong, Jiayi Chen, Han Zhao, Shuanghao Bai, Pengxiang Ding, Pengwei Wang, Haoang Li
- **机构：** The Hong Kong University of Science and Technology (Guangzhou), Xiaomi Robotics, Southeastern University, Tsinghua University, Zhejiang University, Westlake University, Symbiosis Robotics, Peking University, Xi'an Jiaotong University, Beijing Academy of Artificial Intelligence
- **版本 / 载体：** arXiv:2610.07969v1 [cs.CV]（2026-10-06 提交）
- **原文：** <https://arxiv.org/html/2610.07969>（摘要页：<https://arxiv.org/abs/2610.07969>）
- **项目页 / 代码：** 未确认开源代码（arXiv:2610.07969v1 全文及参考文献列表未提供 EmbodiedSmith 官方项目页或独立源码仓库；文中引用的 `Genesis-Embodied-AI/genesis-world` 与 `NVlabs/GR00T-WholeBodyControl` 分别为外部仿真引擎与人形全身控制底座仓库，非本文官方源码发布）。
- **核对范围：** 已通读 arXiv:2610.07969v1 HTML 全文，涵盖 Abstract、§1 Introduction、§2 Related Work、§3 Method（§3.1–§3.3）、§4 Experiments（§4.1–§4.3、Table 1–4、Figure 1–7 图注）、§5 Conclusion、References 及 Appendix §6–§10（Table S1–S6、Figure S1–S2 图注），共 1479 行文本。
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [EmbodiedSmith：仿真中基于递归自改进飞轮的具身数据扩展框架](../../wiki/entities/paper-agentic-2610-07969.md)
- **核心架构核对：**
  - **环境构建（§3.1）：** 包含资产生成（VLM 规范化开放资产 + 图像空间参考图精修 + SAM3D 重建 + 自适应凸分解与物理/接触验证）、物理资产生成（基于 Articraft 迭代构造铰接物体，并支持可变形体与流体及 CuRobo 可行性检查）、场景生成（VLM 生成类型化关系图 + Beam-DFS 分层布局优化与 CuRobo 可达性校验，输出初始场景 $\mathcal{S}_0=(\mathcal{G},\mathcal{O},\mathcal{X},\mathcal{Q})$）以及模式受限的场景编辑（Scene Edit，支持增删、替换、缩放与重定位及可追溯修订记录）。
  - **任务与轨迹生成（§3.2）：** 由 Task Plan Agent（评测使用 `gpt-5.6-sol` high reasoning effort）、Harness 调度器与 `simengine`（定量基准采用 `Isaac Sim 6.0.1`，物理与渲染步频均为 30 Hz）组成；代理通过 `inspect`、`build`、`run`、`revise`、`finish` 五个高层工具把指令映射为 23 种原子技能（Table S1）构成的 Skill DAG，并在程序化验收之外叠加独立 VLM 评审（`inspect`），失败则由执行遥测驱动任务重规划或触发上游 Scene Edit。
  - **多构型扩展（§3.3）：** 通过底层执行适配器（Adapter）与上层可复用 Skill DAG 解耦，支持 Unitree G1 人形机器人（50 Hz 合并 15-DoF 下肢平衡行走策略与 14-DoF 手臂运动学/CuRobo 规划，手部独立控制）以及 Split ALOHA + Sharpa Wave 灵巧手（双臂各 6-DoF + 22-DoF 灵巧手，配合 fail-closed 接触契约与 PhysX 逐连杆接触遥测）。
- **实验核对：**
  - **环境生成（§4.1，Table 1）：** 在 20 个室内场景说明 × 3 次独立生成上对比 Holodeck、LayoutVLM、SAGE、EmbodiedGen V2、SceneSmith；采用 GPT-5.5 四视图感知评分（CF/SL/PP/TF/VQ）与几何指标（COL/NAV/OOB/OPN），EmbodiedSmith 在 CF（7.13±1.48）、SL（7.27±1.41）、PP（7.68±0.83）、VQ（7.32±0.79）、COL（0.29±0.35）、OOB（0.00±0.00）、OPN（0.9875±0.0481）上取得表中最优。
  - **示范生成与消融（§4.2，Table 2–3）：** 16 个任务（10 固定基座 + 6 移动操作，各半短程与长程，含记忆回溯任务），每任务 10 次独立会话、最大 30 轮反馈修订、10 个留出仿真种子评测；平均 ASR 67%（Top-5 77%，平均输出 40,204 tokens），高于 SceneSmith+Planner（17%/23%/137,062）与 EmbodiedGen V2+Planner（21%/27%/91,062）。移除执行反馈（w/o Feedback）使四类任务 ASR 降至 0.0%–4.0%；移除放置校验（w/o Placement Validation）使固定长程任务 ASR 从 71.6% 降至 13.2%。
  - **下游策略训练（§4.3，Table 4、Table S4–S6）：** 书房与厨房 10 个 Franka FR3 任务（A1–A5, B1–B5），对比 20 小时固定布局 `clean` 数据与 10+10 小时 `mixed` 随机化数据训练 $\pi_{0.5}$ 与 FastWAM；在随机化评测（Random）下，$\pi_{0.5}$ 平均成功率由 7.2% 升至 40.2%，FastWAM 由 1.0% 升至 10.2%。
