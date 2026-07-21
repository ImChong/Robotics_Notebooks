# Xiaomi-Robotics-1: Scaling Vision-Language-Action Models with over 100K Hours of Real-World Trajectories

> 来源归档（ingest）

- **标题：** Xiaomi-Robotics-1: Scaling Vision-Language-Action Models with over 100K Hours of Real-World Trajectories
- **类型：** paper（arXiv 预印本）
- **机构：** Xiaomi Robotics（小米机器人实验室）
- **原始链接：**
  - <https://arxiv.org/abs/2607.15330>
  - <https://arxiv.org/html/2607.15330>
  - <https://arxiv.org/pdf/2607.15330>
  - 项目页：<https://robotics.xiaomi.com/xiaomi-robotics-1.html>
  - 技术报告 PDF：<https://robotics.xiaomi.com/robot-static-resource/xiaomi-robotics-1/xiaomi-robotics-1.pdf>
  - 代码占位仓：<https://github.com/XiaomiRobotics/Xiaomi-Robotics-1>
  - 模型组织页：<https://huggingface.co/XiaomiRobotics>
- **发布日期：** 2026-07-16（arXiv v1）
- **入库日期：** 2026-07-21
- **一句话说明：** 小米 **具身基座 VLA（XR-1）**：**>100k h** embodiment-free **UMI** 预训练（**Qwen3.5-27B** 自动标注状态转移语言）+ **~10k h** 跨本体后训练对齐真机与 imperative 指令；**Qwen3-VL + DiT flow matching + Choice Policies** 的 **MoT**（**2.6B / 5.1B / 10.5B**）；预训练 scaling 可预测迁移到未见环境开箱与少样本微调；仿真 **RoboCasa / RoboCasa365 / VLABench / RoboDojo** 四榜领先；**代码与权重仓已建链但截至入库日仍为 Coming soon**。

## 核心论文摘录（MVP）

### 1) 动机：机器人缺 LLM 式 scaling，真机 teleop 是瓶颈

- **链接：** <https://arxiv.org/html/2607.15330#S1>
- **摘录要点：** VLA / WAM 已现「数据越大越强」苗头，但主流 **真机遥操作** 慢、贵、分布窄。XR-1 用 **可规模化的 UMI 无机器人轨迹** 做预训练 breadth，再用跨本体后训练做 embodiment / 指令 alignment。
- **对 wiki 的映射：**
  - [Xiaomi-Robotics-1](../../wiki/entities/xiaomi-robotics-1.md) — 定位与 scaling 主张。
  - [VLA](../../wiki/methods/vla.md) — 大模型 scaling 在动作策略上的实证。

### 2) 架构：MoT = Qwen3-VL + DiT flow matching + Choice Policies

- **链接：** <https://arxiv.org/html/2607.15330#S2>
- **摘录要点：**
  - **VLM** 编码观测与语言；**Choice Policies** 在 VLM 侧预测 **K** 候选 action chunk + 分数，winner-take-all **L1**。
  - **DiT** 层数对齐 VLM、hidden 更小；条件于 **本体状态 + VLM KV**，**flow matching** 生成 chunk；推理 **5 步 Euler**（Δτ=0.2）。
  - **关键：** DiT **不 attend** VLM 的 action-related tokens，防止抄 VLM 动作捷径。
  - **规模：** **2.6B / 5.1B / 10.5B**（VLM **2.1B / 4.4B / 8.8B** + DiT **470M / 604M / 1.5B**）。
- **对 wiki 的映射：**
  - [Xiaomi-Robotics-1](../../wiki/entities/xiaomi-robotics-1.md) — 核心结构表。
  - [Xiaomi-Robotics-0](../../wiki/entities/xiaomi-robotics-0.md) — 同族 Qwen3-VL + DiT + Choice Policies。

### 3) 预训练：>100k h UMI + VLM 自动状态转移标注

- **链接：** <https://arxiv.org/html/2607.15330#S2.SS2.SSS1>
- **摘录要点：** 等长切段 → **Qwen3.5-27B** 描述夹爪/物体 **状态转移**；producer–consumer 管线约 **两周** 标完。损失 \(L_{\text{Flow}}+L_{\text{Regression}}+0.1 L_{\text{NTP}}\)；VL:UMI = **1:9**；每样本 **4** 个 flow 时间步摊销 VLM。
- **对 wiki 的映射：**
  - [Teleoperation](../../wiki/tasks/teleoperation.md) — UMI 无机器人示范范式。
  - [Xiaomi-Robotics-1](../../wiki/entities/xiaomi-robotics-1.md) — 数据与损失。

### 4) 后训练：~10k h 跨本体 + imperative 指令对齐

- **链接：** <https://arxiv.org/html/2607.15330#S2.SS2.SSS2>
- **摘录要点：** **>7.2k h** 移动操作/双臂真机 + **>1k h** 人工 imperative UMI + 过滤 **Bridge V2 / RT-1 / DROID**；臂动作用相对 delta EE；统一 action 向量并对缺失维 **mask**。采样比 VL:开源:UMI:自采 ≈ **0.5:0.5:0.5:8.5**。
- **对 wiki 的映射：**
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md) — 移动操作基座语境。
  - [Manipulation](../../wiki/tasks/manipulation.md) — 后训练任务分布。

### 5) 结果：scaling 迁移、少样本、仿真四榜

- **链接：** <https://arxiv.org/html/2607.15330#S3>
- **摘录要点：**
  - 预训练验证 **MSE** 随数据/模型规模下降；后训练开箱总成功率随预训练规模 **26%→75%**（无 action 预训练 → 100% of 20k h），**2B→10B → 61%→79%**。
  - 少样本四任务（phone packing / printer / laundry / box）：**<10 h/任务** 平均 **75%** vs **π₀.₅ 40%**。
  - 仿真表：RoboCasa **74.5%**；RoboCasa365 平均 **57.4%**（摘要写 57.6%）；RoboDojo 平均分 **20.07** / 成功率 **13.93%**（摘要强调 score 20.07）。
- **对 wiki 的映射：**
  - [Xiaomi-Robotics-1](../../wiki/entities/xiaomi-robotics-1.md) — 实验结果表（以正文表为准，注明摘要/项目页口径差）。

### 6) 开源状态（步骤 2.5 核查，2026-07-21）

- **项目页：** 已链 **arXiv Report**、**GitHub Code**、**Hugging Face Model**。
- **GitHub `XiaomiRobotics/Xiaomi-Robotics-1`：** 仅 **README + PDF**；README 徽章写 **Code / Model Coming soon**，无可运行训练/推理入口。
- **Hugging Face `XiaomiRobotics`：** 组织页存在，但 **XR-1 权重尚未上架**（可见条目多为 XR-0 / U0）。
- **结论：** **宣称将开源 / 仓库占位已建**，截至入库日 **无可运行实现** → wiki「源码运行时序图」写 **不适用**。
- **对 wiki 的映射：**
  - [sources/repos/xiaomi-robotics-1.md](../repos/xiaomi-robotics-1.md)
  - [sources/sites/xiaomi-robotics-1.md](../sites/xiaomi-robotics-1.md)

## 当前提炼状态

- [x] arXiv abs/HTML 与项目页、GitHub、HF 组织页对齐
- [x] wiki 实体页补充 arXiv:2607.15330 与开源边界刷新
- [x] 与 XR-0 / XR-U0 / VLA 交叉引用已存在，本轮补 paper source 链路
