# Closing the Loop in Humanoid VLA: Persistent 3D Object Tokens for Verifiable Loco-Manipulation（arXiv:2607.18016）

> 来源归档（ingest）

- **标题：** Closing the Loop in Humanoid VLA: Persistent 3D Object Tokens for Verifiable Loco-Manipulation
- **类型：** paper
- **来源：** arXiv abs / HTML / PDF
- **原始链接：**
  - <https://arxiv.org/abs/2607.18016>（2026-07-20 提交）
  - HTML：<https://arxiv.org/html/2607.18016v1>
  - PDF：<https://arxiv.org/pdf/2607.18016>
- **作者：** Peng Ren\*, Haoyang Ge\*, Jiang Zhao, Cong Huang, Yukun Shi, Pei Chi†, Kai Chen†（\*Equal contribution；†Corresponding：peichi@buaa.edu.cn / kaichen@zgci.ac.cn）
- **机构：** 北京航空航天大学（BUAA）；北京中关村学院（BZA / Zhongguancun Academy）；天津大学（TJU）；机智赛博（DeepCybo）；中关村人工智能研究院（ZGCI / ZGCA）
- **入库日期：** 2026-07-22（初版）；**深读补全：** 2026-07-26
- **一句话说明：** 针对人形长时程 loco-manipulation 中的 **object-state divergence**，用 Persistent Object Tokenization（POT）从 RGB-D 维护角色索引的持久 3D 对象记录；同一记录同时条件化 GR00T-N1.7 全身动作专家与几何谓词验收，在 Unitree G1 八类任务上由匹配基线 **39/80 → 71/80**。

## 开源状态（核查 2026-07-26）

- **确认未开源：** arXiv abs/HTML 未列项目页、GitHub、Hugging Face 或数据集入口；第三方综述页亦仅链 arXiv。
- **复现边界：** 动作骨干可对照公开 [Isaac GR00T / GR00T-N1.7](../../wiki/entities/isaac-gr00t.md)；POT 的角色化 RGB-D 记录、token sidecar 与谓词监督实现 **无公开代码**。
- **源码运行时序图：** **不适用**（无可运行官方实现）。

## 核心论文摘录（MVP）

### 1) 问题：object-state divergence

- **链接：** <https://arxiv.org/abs/2607.18016> §1
- **摘录要点：** 长时程人形移动操作要求同一杯/篮/支撑面/交接对象在行走、接触、遮挡与恢复后仍可寻址。现代 VLA 通常用当前观测的视觉–语言特征条件化动作，而任务进度由另一套 monitor / 谓词 / 语言状态判定；当 **动作用的对象状态 ≠ 验收用的对象状态** 时，会出现过早子任务切换、错对象操作或无效恢复。根因是物理的：对象在度量 3D 中移动、接触、入容器并被部分遮挡。
- **对 wiki 的映射：**
  - [POT-VLA（论文实体）](../../wiki/entities/paper-pot-vla.md)
  - [VLA](../../wiki/methods/vla.md)
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)

### 2) 方法：POT → Persistent 3D Object Tokens → POT-VLA

- **链接：** arXiv §3.1–3.2
- **摘录要点：**
  - 由 VLM 规划器或人工任务文件给出 typed-subtask 计划；每个子任务指定角色、grounding 查询、成功谓词、失败处理、chunk horizon、超时与重试预算。
  - 在线/缓存 **SAM3** 掩码 + 深度反投影到 **机器人 base 系**，得到角色索引记录（角色、短语、2D box、3D 质心与外延、置信度、可见性、关系特征）。
  - Token schema 默认 **K=8 slots × F=33 features**；角色含 TARGET / DESTINATION / SUPPORT / HANDOVER_PARTNER；遮挡时槽位保留但标不确定。
  - 动作专家实例化为 **GR00T-N1.7**；对象 token 经 LayerNorm→Linear→GELU→Dropout→Linear 投影后插入 DiT action-head 自注意力序列 \(S_t=[Z_t^{\mathrm{state}},Z_t^{\mathrm{obj}},Z_t^{\mathrm{action}}]\)，视觉–语言骨干仍作 cross-attention。
  - 训练沿用基座 action-chunk 损失；演示配对同一 schema 的 object-token sidecar；**无**额外对象记忆或谓词损失。
- **对 wiki 的映射：**
  - [POT-VLA](../../wiki/entities/paper-pot-vla.md)
  - [Isaac GR00T](../../wiki/entities/isaac-gr00t.md) / [GR00T N1](../../wiki/entities/paper-hrl-stack-34-gr00t_n1.md)
  - [Unitree G1](../../wiki/entities/unitree-g1.md)

### 3) 闭环：几何谓词监督与恢复

- **链接：** arXiv §3.3
- **摘录要点：** 谓词监督与动作专家共享刷新后的对象记忆 \(\mathcal{M}^{\tau_i}_{t+H}\)。谓词覆盖 containment、support、末端–目标邻近、目标–目的地位移、双臂分配、交接距离等；状态 ∈ {in_progress, done, blocked, failed, uncertain}。低置信触发 reobserve/reground；诊断如 object_not_grasped / outside_goal_region 再驱动 retry 或 replan。几何谓词优先；可选 VLM 处理语义歧义。
- **对 wiki 的映射：**
  - [POT-VLA](../../wiki/entities/paper-pot-vla.md)
  - [Being-0](../../wiki/entities/paper-loco-manip-161-057-being-0.md) — 外部服务任务对照

### 4) 主结果、消融与泛化

- **链接：** arXiv §4；Table 1–2
- **摘录要点：**
  - **平台：** Unitree G1 + Dex3-1 + 头戴 RGB-D；桌面机跑感知/token/谓词，机载跑低层臂与行走；每任务 10 次真机 trial。
  - **匹配对照（同 GR00T-N1.7 / 同 embodiment / 同 runtime）：** Direct **39/80** → POT-VLA **71/80**。最大增益：叠三杯 **1/10→8/10**、衣物入篮 **3/10→9/10**、双球入篮 **5/10→9/10**、抽屉/托盘 **4/10→8/10**。
  - **Being-0 对齐服务任务（外部文献对照，非本地复现）：** POT-VLA **44/50** vs Being-0 报告 **37/50**。
  - **消融（四高发散任务，40 trial/变体）：** Direct 15/40；Verifier only 22/40；POT tokens only 31/40；全量 34/40 → **token 条件化贡献最大，谓词捕获残余假完成**。
  - **对象状态扰动：** 新实例 6→9、布局移位 5→9、干扰物 8→9、执行中扰动 4→8（各 /10）。
- **对 wiki 的映射：**
  - [POT-VLA](../../wiki/entities/paper-pot-vla.md)
  - [VLA](../../wiki/methods/vla.md)

## 对 wiki 的映射（汇总）

- [`wiki/entities/paper-pot-vla.md`](../../wiki/entities/paper-pot-vla.md) — 主实体页
- [`wiki/methods/vla.md`](../../wiki/methods/vla.md) — 可验证对象中心闭环执行
- [`wiki/tasks/loco-manipulation.md`](../../wiki/tasks/loco-manipulation.md) — 人形移动操作任务
- [`wiki/entities/unitree-g1.md`](../../wiki/entities/unitree-g1.md) — 实机平台
- [`wiki/entities/isaac-gr00t.md`](../../wiki/entities/isaac-gr00t.md) — GR00T-N1.7 动作骨干
- [`wiki/entities/paper-loco-manip-161-057-being-0.md`](../../wiki/entities/paper-loco-manip-161-057-being-0.md) — Being-0 外部对照
