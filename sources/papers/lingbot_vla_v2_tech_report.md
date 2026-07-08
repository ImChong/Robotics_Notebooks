# LingBot-VLA 2.0：From Foundation to Application（技术报告）

> 来源归档（ingest）

- **标题：** From Foundation to Application: Improving VLA Models in Practice（LingBot-VLA 2.0）
- **类型：** paper（技术报告 / 项目 PDF）
- **arXiv abs：** <https://arxiv.org/abs/2607.06403>
- **PDF（仓库内链）：** <https://github.com/robbyant/lingbot-vla-v2/blob/main/assets/LingBot_VLA_2_0.pdf>
- **项目页：** <https://technology.robbyant.com/lingbot-vla-v2>
- **代码：** <https://github.com/robbyant/lingbot-vla-v2>
- **权重：** [Hugging Face `robbyant/lingbot-vla-v2-6b`](https://huggingface.co/robbyant/lingbot-vla-v2-6b) / [ModelScope](https://modelscope.cn/models/Robbyant/lingbot-vla-v2-6b)
- **机构：** Robbyant（蚂蚁集团具身智能团队）
- **规模：** **LingBot-VLA 2.0-6B**（**Qwen3-VL-4B-Instruct** 骨干 + **MoE action expert** + **native depth** 蒸馏分支）
- **数据：** 约 **60,000 h** 预训练混合：**50,000 h** 跨 **20** 种机器人配置的轨迹 + **10,000 h** egocentric 人操作视频
- **入库日期：** 2026-07-08
- **一句话说明：** 相对 LingBot-VLA 1.0，2.0 用 **重设计数据管线**、**55 维统一动作空间**（臂/EEF/夹爪/灵巧手/腰/头/底盘）与 **Dual-Query 深度/视频蒸馏** 把 VLA 从大规模预训练推向 **跨任务/跨本体真机应用**；开源 **6B 权重**、RoboTwin 后训练范例与真机部署脚本。

## 核心摘录（面向 wiki 编译）

### 1) 预训练数据管线（robotic + egocentric 双流过滤）

- **链接：** <https://technology.robbyant.com/lingbot-vla-v2> / README「Pre-Training Data」
- **摘录要点：**
  - 原始池拆为 **机器人轨迹流** 与 **egocentric 人视频流**，分阶段剔除低质量样本。
  - **机器人侧：** 去除 video–state 不对齐、模糊/遮挡/掉帧、多视角错位、速度/加速度/jerk 异常、静态信号占比 **>95%** 等片段。
  - **人视频侧：** VLM 筛 manipulation-centric 片段；重建并标准化 **手部轨迹**；剔除有效手帧 **<20%**、SLAM/相机运动不稳、轨迹不连续等。
  - 覆盖 **单臂、双臂、半身人形、全身人形** 与 egocentric 源，最终约 **6 万小时** 高质量混合。
- **对 wiki 的映射：**
  - [LingBot-VLA 2.0](../../wiki/entities/lingbot-vla-v2.md) — 数据管线 Mermaid 与工程过滤清单
  - [HumanNet](../../wiki/entities/humannet.md) — 同类「人视频小时 vs 真机小时」预训练对照语境
  - [VLA](../../wiki/methods/vla.md) — 异构数据质量对齐实践

### 2) 55 维统一动作表示 + MoE Action Expert

- **链接：** README「Unified Action Representation」「MoE Action Expert」
- **摘录要点：**
  - 将 **20 种 embodiment** 映射到固定 **55 维** 规范向量：臂关节 **14**、EEF 位姿 **14**、夹爪 **2**、手关节 **12**、腰 **4**、头 **2**、移动 **3**、保留 **4**。
  - **Action expert** 内嵌 **稀疏 MoE**：细粒度专家分段 + **shared expert** 隔离，使通用先验与专精 embodiment/任务模式在 **相同激活参数量** 下共存。
  - 后训练配置示例启用 **sequence-wise auxiliary loss** 与 **router z-loss** 稳定 MoE 路由；可切换 **Muon** 或 **AdamW** 优化器。
- **对 wiki 的映射：**
  - [LingBot-VLA 2.0](../../wiki/entities/lingbot-vla-v2.md) — 统一动作空间表与 MoE 机制
  - [Green-VLA](../../wiki/entities/paper-greenvla-staged-vla-humanoid.md) — 另一条 **语义槽位统一动作** 路线对照

### 3) Dual-Query Distillation（LingBot-Depth + DINO-Video）

- **链接：** README「Dual-Query Distillation」
- **摘录要点：**
  - 在视觉/文本 token 后追加 **当前与未来感知 query**。
  - 从 **LingBot-Depth** 蒸馏 **几何/深度** 线索，从 **DINO-Video** 蒸馏 **语义时序** 先验。
  - 目标：让因果推理同时捕获 **当前场景几何** 与 **未来场景演化**，作为 **预测性动力学** 代理任务。
  - 训练还需 **MoGe-2-vitb-normal** 等教师权重（见 `Training_Config.md`）。
- **对 wiki 的映射：**
  - [LingBot-VLA 2.0](../../wiki/entities/lingbot-vla-v2.md) — Dual-Query 流程图
  - [Being-H0.7](../methods/being-h07.md) — 另一类「未来观测监督」潜空间先验对照

### 4) 后训练、评测与真机部署

- **链接：** README「Post-Training Example」「Evaluation and Deployment」
- **摘录要点：**
  - 后训练三步：**LeRobot v2.1/v3.0 数据** → **robot config YAML**（特征映射到统一空间）→ **norm statistics**。
  - 范例：**RoboTwin 2.0 50 任务**（clean + randomized）联合后训练；真机配置见 `real_robot.yaml`（**native depth**）。
  - **开环评测：** `open_loop_eval.py`；**RoboTwin 闭环：** `start_robotwin_infer_and_eval.sh`；**真机：** `deploy.lingbot_vla_v2_policy`。
  - **RTX 4090D** 上 **10 步去噪** 单次推理约 **130 ms**（`--use_compile`）。
- **对 wiki 的映射：**
  - [Manipulation](../../wiki/tasks/manipulation.md) — LeRobot 数据与双臂后训练语境
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md) — 长程移动操作评测

### 5) 公开基准（generalist 联合训练）

- **链接：** README「Performance」/ 项目页「Real-World Benchmark Results」
- **摘录要点：**
  - **GM-100 双臂桌面（progress / success）：** AgileX Cobot Magic **66.2 / 34.4**（优于 π₀.₅ **59.1 / 32.2**、LingBot-VLA 1.0 **58.2 / 30.0**、GR00T N1.7 **36.3 / 17.8**）；Galaxea R1Pro progress **34.6**（success **15.6** 与 1.0 持平）。
  - **长程移动操作：** Astribot S1 冰箱分拣 in-domain **77.1 / 60.0**、OOD **37.0 / 13.3**；Cobot Magic-ARX X5 炉灶清洁 in-domain **84.3 / 66.7**、OOD **67.5 / 40.0**——均高于 π₀.₅ 对照。
- **对 wiki 的映射：**
  - [LingBot-VLA 2.0](../../wiki/entities/lingbot-vla-v2.md) — 结果表与 generalist 设定说明
  - [VLA](../../wiki/methods/vla.md) — π₀.₅ / GR00T 同赛道索引

## 当前提炼状态

- [x] GitHub README + 项目页 + arXiv 元数据已对齐摘录
- [x] wiki 映射：`wiki/entities/lingbot-vla-v2.md` 新建，并与 VLA / HumanNet / Green-VLA 交叉引用
- [ ] 待社区复现后补独立 arXiv HTML 细读与消融表
