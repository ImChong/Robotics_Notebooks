# MessyMem: Learning-from-Doing Memory for Mobile Manipulation

- **作者：** Anuva Banwasi, William Muckelroy III, Priya Sundaresan, Linfeng Zhao, Jeannette Bohg, Cherie Ho
- **机构：** Stanford University
- **版本 / 载体：** arXiv:2609.15976v2 [cs.RO]（Submitted 14 Sep 2026, v2 15 Sep 2026；Accepted at CoRL 2026；26 pages, 8 figures）
- **原文：** [arXiv HTML 全文 (2609.15976v2)](https://arxiv.org/html/2609.15976) · [arXiv 摘要页](https://arxiv.org/abs/2609.15976) · [官方项目页](https://messymem.github.io)
- **项目页 / 代码：** 宣称将开源或未列链接（官方项目页 `https://messymem.github.io` 标明 `Code: soon`、`Video: soon`，截至 2026-11-27 尚未发布官方代码仓库；页面源码所引 `nerfies/nerfies.github.io` 仅为网页模板，非本文算法实现）
- **核对范围：** 已通读 arXiv HTML 全文（Abstract、§1–§6、References 及 Appendix A–D 全部内容，含 Table 1–4、Figure 1–8 图注、Interaction Analyzer / Retriever / Planner 完整 Prompt 与 JSON Schema）以及项目页 `https://messymem.github.io` 全文。未发现官方可运行代码仓库，故未进行静态源码核对。
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [MessyMem: Learning-from-Doing Memory for Mobile Manipulation](../../wiki/entities/paper-agentic-2609-15976.md)
- **核心问题与设计目标：** 移动操作机器人跨房间、跨任务长期部署时，需要同时满足三项记忆属性：（1）全局可检索与空间接地（globally queryable & spatially grounded）；（2）可从物理交互中更新（updatable from interactions），如锁闭状态、重量、可抓取性与容器内部物体；（3）细粒度视觉可追溯（fine-grained），保留文本摘要丢失的外观标签与杂乱度细节。
- **三元记忆架构（§3, Appendix A–B）：**
  1. **3D 场景图（SG）：** 使用 SAM3（封闭词表提示词，见 Table 3）对 RGB-D 帧进行实例分割并反投影至 3D，按语义标签与空间邻近性聚类融合为持久化物体/固定装置节点（含 `pos`、`bbox_3d`、`properties`、`contents`、`task_history`），节点离开视野后仍保留在图中。
  2. **VLM 交互分析器（IA）：** 每次操作原语执行后，读取操作前（BEFORE）、操作后（AFTER）全分辨率图像与交互窗口多视角时间拼贴图（collages，涵盖 wrist / side / front 视角），区分“抓空/滑脱（failed grasp，可重试）”与“抓牢但未动（locked/stuck，应切换容器）”，输出严格 JSON 更新 `accessible`、`pickable`、`pushable`、`fullness`、`openable`（及 `is_opened`）、开放词汇属性与 `contents`（新证据覆盖旧结构化状态）。
  3. **关联关键帧记忆（KF）：** 保存交互前后及信息量峰值帧（`interaction_pre` / `during` / `post` / `peak`）与场景图显著变化帧（`scene_graph_delta`），仅对感知触发帧做空白/模糊/近似重复过滤，交互帧全部保留；存储全分辨率图、缩略图及关联场景图节点元数据。
  4. **两阶段检索与闭环规划（§3.4, Appendix B.3）：** 从场景图关联帧、标签匹配帧和近期交互帧三路按上限聚合候选池；先由纯文本 VLM 依据目录元数据粗筛 `max_k` 帧，再由多模态 VLM 审阅缩略图精挑细选后传给闭环 VLM 规划器（每次只执行第一步再重规划；真机基座帧叠加 Set-of-Mark 黄色圆圈投影节点 ID）。
- **核心实验数据（§4–§5, Appendix C–D）：**
  - **仿真（RoboCasa365 / MuJoCo, Franka Panda + Omron 底盘，每任务 12 次规划器调用上限，每方法 50 次实验报告任务进度 % 及 95% CI）：**
    - *Cluttered Pick*（两柜均有番茄酱，6 件宽松 vs 3 件拥挤）：MessyMem **84.0%** `[77.0, 90.0]` vs SG+IA 58.0%、RoboEXP 53.0%、SG+KF 51.0%、SG 49.0%、MemER+Fixtures 8.0%、MemER-style 0.0%。
    - *Locked / Unlocked Cabinets*（三柜含 1 锁柜，先后找香蕉和番茄酱）：MessyMem **99.0%** `[97.0, 100.0]` vs SG+IA 88.0%、RoboEXP 70.0%、SG 50.0%、SG+KF 43.0%、MemER+Fixtures 27.0%、MemER-style 0.0%。
    - *25-Task Long Horizon*（10 柜 60+ 物体、3 锁柜、连续 >3 小时不重置记忆，含 Finding 9 / Matching 9 / Manipulation 3 / Memory queries 4）：MessyMem **80.0%** `[77.4, 82.6]`（显著高于 SG+IA 65.2% [+14.8 pp] 与 RoboEXP 51.1% [+28.9 pp]，6 项成对比较均满足 $p_{\mathrm{Holm}} < 10^{-4}$；SG 49.1%、SG+KF 47.9%、MemER-style 1.8%、MemER+Fixtures 0.7%）。长程中位积累 **9,669** 帧关键帧，中位检索延迟 **8.6 s**，中位回溯时长 **62.7 min**。
  - **真机（TidyBot++ 平台：6-DoF YAM 臂 + 全向底盘 + 双 ZED 2 + 腕部 Arducam 鱼眼，DMP 技能 + VLM 3D 点选，人工安全确认）：**
    - *Office Drawer Search*（$n=5$，3 个顺序目标）：MessyMem 唯一全通，整体任务进度 **1.00**、总步数 **10.8**、SPL **0.92**（对比 SG+IA 0.87 / 11.0 / 0.82，SG+KF 0.60 / 17.0 / 0.40，SG 0.40 / 15.4 / 0.34）。
    - *Sock Pairing*（$n=6$，外观颜色/花纹随机配对）：MessyMem 任务进度 **1.00**，对比 SG+IA 0.67、SG+KF 0.67、SG 0.50。
- **局限与边界（§6, Appendix A）：** 感知端采用 SAM3 封闭词表提示（真机故意省略 `scissors` 和 `gaming controller` 以验证封闭词表局限）；交互分析仅建模机器人主动动作，未涵盖人类操作（HRI）；技能层依赖预设脚本或遥操作拟合的 DMP 原语（仿真 `PickPlaceCabinetToCounter` 使用物体真值位姿，真机执行需人工安全把关）。
