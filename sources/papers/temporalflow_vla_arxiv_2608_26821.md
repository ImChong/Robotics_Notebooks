# TemporalFlow-VLA（物理接地执行历史）

> 来源归档（ingest）

- **标题：** TemporalFlow-VLA: Learning Physically Grounded Execution History for Long-Horizon Robot Manipulation
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.26821>
- **机构：** 香港科技大学（广州）；浙江大学；西蒙菲莎大学（SFU）；智元机器人（AgiBot）
- **入库日期：** 2026-08-31
- **一句话说明：** 在 π₀.₅ 上并行学习两个 chunk 对齐时序 query（Q₈/Q₁₅），用离线机器人表面时序流（关节+URDF+标定相机）作训练期物理监督；部署无几何管线，异步特征缓存维持单帧级采样延迟。

## 核心摘录（MVP）

### 1) 多帧历史≠可用执行历史

- **摘录要点：** 单纯堆叠历史帧无法可靠编码近期物理变化；多阶段操纵中视觉相似状态需依执行史区分动作。诊断：打乱 t−15/t−8 两帧顺序几乎不改变 flow-matching 损失，而去掉历史约 +4.6%。
- **对 wiki 的映射：**
  - [TemporalFlow-VLA](../../wiki/entities/paper-temporalflow-vla.md) — 问题动机与历史顺序敏感性。
  - [π₀.₅](../../wiki/entities/paper-pi05-open-world-vla.md) — 基座 VLA。

### 2) 机器人表面时序流 + 分层时序 query

- **摘录要点：** 离线用关节、URDF、相机标定将机器人表面像素投影为 16×16×2 的区间流场（ρ∈{8,15}，对齐 16 步 action chunk）。两 learnable query Q₈（t−8→t）与 Q₁₅（t−15→t，可读 Q₈）经 directed masked attention 压缩历史；动作 token 只能经 Q₈/Q₁₅ 接触历史，不能直接读历史 patch。训练叠加 Huber 流重建损失 L_temp（λ=1.0）与原 L_action。
- **对 wiki 的映射：**
  - [TemporalFlow-VLA](../../wiki/entities/paper-temporalflow-vla.md) — 方法与注意力掩码。
  - [VLA](../../wiki/methods/vla.md) — 长程记忆增强脉络。

### 3) 仿真与 AgiBot A3 真机

- **摘录要点：** LIBERO 四套件联合 30k 步：平均 **97.63±0.26%**，Long **96.60±0.87%**。RoboTwin 2.0 十二任务 60k 步：Clean/Randomized **85.5%/84.2%**，H=3 子集优势最大（Rand. **87.5%**，较次优 +14.5 pt）。真机三阶段叠杯 **57.8%→77.8%**、双瓶装箱 **86.7%→97.8%**。异步历史特征缓存：LIBERO Long 采样延迟 **68.10→62.78 ms**（−7.8% 累计）。
- **对 wiki 的映射：**
  - [TemporalFlow-VLA](../../wiki/entities/paper-temporalflow-vla.md) — 指标读法。
  - [Manipulation](../../wiki/tasks/manipulation.md) — 长程双臂操纵语境。

### 4) 开源状态（截至 2026-08-31）

- **摘录要点：** arXiv 正文未列项目页或 GitHub；网络检索亦无官方仓库 → **截至入库日未列官方代码或权重 URL**。
- **对 wiki 的映射：**
  - [TemporalFlow-VLA](../../wiki/entities/paper-temporalflow-vla.md) — 局限节标明待发布。

## 当前提炼状态

- [x] arXiv 摘要与正文方法/实验节已对齐摘录
- [x] wiki 映射：`wiki/entities/paper-temporalflow-vla.md` 新建
