# DEUX（XYZ Corp · 半人形服务机器人产品页）

- **类型：** 产品 / 项目站（商业服务机器人）
- **收录日期：** 2026-07-27
- **产品页：** <https://xyzcorp.imweb.me/DEUX>
- **公司主站：** <https://xyzcorp.imweb.me/>
- **公司 ABOUT：** <https://xyzcorp.imweb.me/93>
- **TECHNOLOGY：** <https://xyzcorp.imweb.me/tech>
- **联系：** contact@xyzcorp.io
- **说明：** 韩国 **XYZ（(주)엑스와이지 / XYZ Corp）** 的 **半人形（semi-humanoid）双臂移动服务机器人 DEUX 1.0** 产品页；配套 **Glove X** 数据采集手套与 **Brain X** 行为/智能栈。

## 一句话

面向零售/办公/医院/家庭的 **真店数据驱动 Physical AI 服务机器人**：三指手 + 1:1 Glove X 零样本重定向采集，Brain X 持续学习；官网宣称含 BrainX 与一轮任务建模的预购价位。

## 为什么值得保留

- **产业侧「三指 + 手套 1:1」样本**：与 [Sunday ACT-2 / Memo](../../wiki/entities/sunday-robotics-act2.md) 的家务三指叙事并列，但强调 **手套–手接触点对齐 → 免后处理 retarget** 的采集飞轮。
- **移动双臂服务机器人规格可引用**：高度可调、万向移动基座、CAN-FD 1 kHz、ROS 2 / Python SDK 叙事；便于与开源/学术平台对照选型。
- **闭源商业 Physical AI 闭环案例**：真店数据 → Glove X 多模态采集 → Brain X（IL/RL/对话动作）→ 门店试点，适合写进遥操作/模仿学习产业对照，而非复现入口。

## 项目页核查（步骤 2.5 · 2026-07-27）

| 核查项 | 结论 |
|--------|------|
| **DEUX 页导航 / Footer** | ABOUT / TECHNOLOGY / ROBOT / CONTACT、Career、SNS（YouTube / Facebook / Medium）；**无 Code / GitHub / Hugging Face / Zenodo 入口** |
| **HTML 检索** | `github.com` / `huggingface` / `open-source` **无匹配**；可见 **ROS 2 Support**、**NVIDIA Thor（选配另售）** |
| **同名 GitHub 组织** | `XYZCorp` / `xyz-corp` 等公共仓数为 **0**，且 **未** 从产品页链出；**不得**当作官方开源入口 |
| **开放程度** | **未开源** — 训练/推理代码、权重、真店数据集、完整 CAD/固件均未公开 |
| **部分开放** | 营销规格表、预购价、场景视频/动图；软件侧仅宣称 **ROS 2 / Python / DEUX Controller**（无公开仓） |

- **代码：** 截至入库日 **无官方仓库链接**
- **数据集：** **未公开**（真店 / Glove X 多模态数据为 proprietary）
- **模型 checkpoint：** **未公开**（Brain X / behavior models）

## 公开信息要点（产品页归纳）

### 定位与场景

| 项 | 内容 |
|----|------|
| 产品名 | **DEUX**（半人形服务机器人；ABOUT 时间线称 **DEUX 1.0**） |
| 叙事 | *Physical AI trained on real-store data*；「从开店到打烊」零售工作流自动化 |
| 场景 | **Retail**（补货、清桌、开闭店）、**Office**（清桌、摆椅、递文件）、**Hospital**（陪护/移动辅助叙事）、**Home**（家务） |
| 配套 | **Glove X**（1:1 配对数据采集）、**Brain X**（行为模型 + agentic AI） |

### 四要素（产品页）

1. **User-Centered Design** — 全感官交互设计  
2. **Service robot** — 日常空间共处  
3. **Glove X** — Physical AI 数据采集装置，与 DEUX **1:1 配对**  
4. **Brain X** — 行为模型与持续进化智能  

### 机器人规格（产品页 Spec 表）

| 字段 | 宣称值 |
|------|--------|
| 外形尺寸 | W **530** × D **652** mm |
| 高度 | **900–1550** mm（可调） |
| 重量 | Base **35** kg / Robot **25** kg / Total **60** kg |
| 关节 | 表头写 **Total 30 DoF**，分项为 **7DoF 臂×2 + 7DoF 灵巧手×2 + 1DoF 升降 + 3DoF swerve 移动基座**（分项合计 **32**；营销文案另写 **32 DoF** — **以分项与营销 32 为准，表头 30 疑为笔误**） |
| 电池 | **24 V / 60 Ah（1,440 Wh）** |
| 臂展 / 工作范围 | Range of Motion **501** mm |
| 负载 | 单臂 **5.5** kg / 双臂 **11** kg |
| 控制 | **1,000 Hz CAN-FD** |
| 软件 | **ROS 2**、Python、DEUX Controller |
| 算力 | **NVIDIA Thor 另售（optional）** |
| 移动 | **360°** 全向；交互强调自然连接 |

### Glove X（采集装置）

| 能力 | 宣称 |
|------|------|
| 形态 | 独立系统：**无需外接 PC / 线缆**；板上 **1 kHz** 高速数据处理 |
| 同步架构 | 单 MCU 同步：**7** 关节角 + **3** 指尖压力 + **2** 路视觉；全链路延迟 **&lt; 50 ms** |
| 视觉 | 高分辨率双相机 + **220°** 超广角；**MIPI CSI-2** |
| 关节跟踪 | 磁编码器 **7-DoF @ 1,000 Hz**，精度公差 **0.5°**；可与 **Meta Quest** 双手 3D 跟踪结合 |
| 触觉 | **3** 通道指尖压力，**83.3 Hz** |
| 重定向 | **Zero-shot retargeting**：手套与机器人手接触点 **1:1**，宣称 **无需后处理校正** 即可供模仿学习 |

### Brain X（TECHNOLOGY 页摘要）

- 机器人动作模型框架：融合 **robot foundation models + RL + IL**  
- **System 1 / System 2** 按任务复杂度切换；System 2 为多模态机器人 agent（感知/推理/人机交互）  
- 层级/组合式 RL 编排子策略；**Voice X** 对话动作模型；人类运动模仿 / 视频 retarget 叙事  
- 另有 **TwinX**（Unity 实时数字孪生遥操作）、**Glass X**（眼镜式采集）等生态组件（本 ingest 以 DEUX 页为主，仅作交叉备注）

### 预购价（产品页 Pre-order，USD）

| SKU | 标价 |
|-----|------|
| Mobile DEUX | **$39,900** |
| DEUX | **$29,900** |
| Glove X（每只） | **$3,900** |

说明：含 **BrainX + 一轮任务建模**；**NVIDIA Thor 算力另售**。

### 公司时间线（ABOUT，与 DEUX 相关）

| 日期 | 事件 |
|------|------|
| 2026-04 | 发布双臂半人形 **DEUX 1.0** |
| 2026-07 | **DEUX 1.0** 启动商业门店试点部署 |
| 2026-01 | Series B **KRW 130 亿**；合办 ROSCon Korea 2026 |
| 2026-06 | 与 **NVIDIA** Physical AI Training Program（AI Campus）合作等 |

## 交叉链接

- Wiki 实体：[xyz-deux.md](../../wiki/entities/xyz-deux.md)
- 对照：[sunday-robotics-act2.md](../../wiki/entities/sunday-robotics-act2.md)、[handumi.md](../../wiki/entities/handumi.md)
- 任务 / 对比：[teleoperation.md](../../wiki/tasks/teleoperation.md)、[data-gloves-vs-vision-teleop.md](../../wiki/comparisons/data-gloves-vs-vision-teleop.md)
