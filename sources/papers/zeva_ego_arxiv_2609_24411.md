# Zeva-Ego（第一人称 mid-training + ICCL 部署进化）

> 来源归档（ingest）

- **标题：** Zeva-Ego: Egocentric Mid-Training with In-Context Causal Learning for Robot Manipulation
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2609.24411>
- **机构：** 清华大学 AIR；Z-Trans AI
- **项目页：** <https://air-embodied-brain.github.io/Zeva-Ego/>
- **代码：** <https://github.com/air-embodied-brain/Zeva>（分支 **`feature/zeva_ego`**）
- **入库日期：** 2026-09-29
- **一句话说明：** 从 π0.5 出发，用 ACE 将 10K+ 小时 egocentric 视频转为相机系 action-token 监督做 VLA mid-training，再冻结 VLM、仅对 Action Expert 注入 ICCL 因果上下文；RoboTwin Hard 上 10K Ego 约等同 2K 机器人示教，部署四次尝试 ICCL 可将成功率从 58% 提到 89%。

## 核心摘录（MVP）

### 1) Action-Centric Encoder（ACE）与 egocentric mid-training

- **摘录要点：** ACE 从 RGB 帧对导出连续 action-token 监督；统一末端执行器表示与相机系 action chunk，减弱人腕—机器人 EEF 几何鸿沟；有标注与无标注 egocentric 视频混合 mid-training（伪标签连接观测、相机系动作与子任务语言）。
- **对 wiki 的映射：**
  - [Zeva-Ego](../../wiki/entities/paper-zeva-ego.md)

### 2) ICCL 后训练与冻结分层

- **摘录要点：** 继承 [Zeva](./zeva_arxiv_2608_30880.md) 的 ICCL：CTE 编码视觉状态、已执行动作与观测反馈；BIT 存单次尝试内证据，PIM 跨尝试累积；因果上下文 **仅注入 Action Expert**，高层 VLM 子任务通路不变；部署期 **全参数冻结**，只更新交互记忆。
- **对 wiki 的映射：**
  - [Zeva-Ego](../../wiki/entities/paper-zeva-ego.md)
  - [Zeva](../../wiki/entities/paper-zeva.md)

### 3) 数据效率与评测

- **摘录要点：** 同 π0.5 初始化下，Ego 从基线扩到 **10K 小时** 将 RoboTwin 成功率 **63.8% → 75.3%**，接近 **2K 小时** 机器人示教的 **74.7%**（经验比约 **4–5 : 1** Ego:Robot）；ACE 在 EgoDex–AgiBot 转移上 token 距离与物理动作距离相关 **0.801**，零样本 EgoVerse **0.724**；ICCL 四次尝试 **58% → 89%**（无参数更新）。
- **对 wiki 的映射：**
  - [Zeva-Ego](../../wiki/entities/paper-zeva-ego.md)
  - [Zeva-Ego 项目页](../sites/zeva-ego.md)

### 4) 开源与复现入口

- **摘录要点：** 项目页链至 GitHub；**Zeva-Ego 实现位于 `air-embodied-brain/Zeva` 的 `feature/zeva_ego` 分支**（含 CTE/EAP/PIM、RoboTwin 四阶段训练脚本、`pipelines/ego_action_encoder` 与 `robotwin_clean`）。截至入库日项目页 **未单独列出 HF 权重**（与原版 Zeva RoboCasa  checkpoint 区分）。
- **对 wiki 的映射：**
  - [air-embodied-brain/Zeva（Zeva-Ego 分支）](../repos/air-embodied-brain-zeva.md)
  - [Zeva-Ego 项目页](../sites/zeva-ego.md)

## 当前提炼状态

- [x] 项目页开源核查（代码在 `feature/zeva_ego`）
- [x] wiki 映射：`wiki/entities/paper-zeva-ego.md` 新建
