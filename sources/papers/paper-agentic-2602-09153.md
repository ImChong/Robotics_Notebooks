# SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes

- **作者：** Nicholas Pfaff, Thomas Cohn, Sergey Zakharov, Rick Cory, Russ Tedrake
- **机构：** Massachusetts Institute of Technology (MIT), Toyota Research Institute (TRI)
- **版本 / 载体：** arXiv:2602.09153v2 [cs.RO]（2026-05-30 修订，v1 提交于 2026-02-09）；ICML 2026 Spotlight（Top 2.2%）
- **原文：** https://arxiv.org/abs/2602.09153（全文核对使用 arXiv HTML 版 https://arxiv.org/html/2602.09153）
- **项目页 / 代码：** https://scenesmith.github.io/ | https://github.com/nepfaff/scenesmith（MIT License，已开源：包含五阶段智能体管道、资产路由与多 GPU 生成服务端、Drake 物理后处理、Blender 渲染服务与 Docker 配置；预处理 ArtVIP 数据集托管于 HuggingFace `nepfaff/scenesmith-preprocessed-data`）
- **核对范围：** 通读 arXiv:2602.09153v2 全文（Abstract、Sections 1–5、Appendices A–S、Tables 1–19、Figures 1–34 题注）及项目页 https://scenesmith.github.io/ 公开展示文本，并静态核对官方仓库 https://github.com/nepfaff/scenesmith 的目录结构与 README。
- **通读：** 已完成

## 原文核对笔记

对 wiki 的映射：[SceneSmith 实体页](../../wiki/entities/paper-agentic-2602-09153.md)

1. **核心问题与场景表示**
   - 现有室内场景合成方法（ProcTHOR、Holodeck、I-Design、LayoutVLM、SceneWeaver、HSM 等）主要聚焦家具级布局与视觉真实感，物体数量稀疏（平均 11–23 个/房间），缺乏高密度可操作小物、关节家具以及仿真所需的碰撞几何和物理参数。
   - SceneSmith 将场景表示为房间集合 $\mathcal{S} = \{\mathcal{R}_j \mid j=1,\dots,M\}$，每个房间 $\mathcal{R}_j = (\mathcal{G}_j, \mathcal{O}_j)$ 包含体素/有限厚度建筑几何 $\mathcal{G}_j$ 与仿真就绪物体集合 $\mathcal{O}_j = \{(\mathcal{A}_i, \mathcal{X}_i)\}$，其中 $\mathcal{A}_i$ 包含视觉网格、V-HACD 凸碰撞几何、物理属性（质量、质心、惯量张量、摩擦系数）及关节定义，$\mathcal{X}_i \in SE(3)$。

2. **五阶段层次构建与 Designer–Critic–Orchestrator 三元交互**
   - 构建树包含五个顺序阶段：建筑布局（Layout）→ 落地家具（Furniture）→ 墙面物体（Wall-mounted）→ 天花板物体（Ceiling-mounted）→ 可操作小物（Manipulands）。后三类家具/墙/天花板阶段按房间并行分支，小物阶段按支撑实体（桌面、书架层、地面区域等）进一步分支并行，并伴随层次化 Prompt 细化（$\mathcal{T} \to \mathcal{T}_j \to \mathcal{T}_{j,k}$）。
   - 每阶段均由三个 VLM 智能体（论文配置全部使用 GPT-5.2，Designer/Critic 为 high reasoning effort，Orchestrator/Planner 为 low，辅助 VLM 任务为 medium）协作完成：Designer 调用工具修改场景，Critic 仅拥有观测与验证工具并给出 0–10 分类评分及文字建议，Orchestrator 控制迭代、维护检查点并在评分下降时回滚（保留上下文记忆但恢复场景状态）。
   - 会话记忆采用滑动窗口（保留当前与上一轮完整交互，更早轮次由 LLM 摘要，视觉渲染按需释放）；智能体每轮可调用任意数量工具并组合结构化状态查询（`get_current_scene_state` / `render_ascii`）与带坐标网格/方向箭头的视觉渲染（`observe_scene`）。

3. **资产获取与路由（Asset Router）**
   - **阶段过滤与复合拆解**：拒绝错配阶段的资产请求，并将“fruit bowl”“desk with monitor”等复合请求拆分为独立可操作原子资产。
   - **三路获取策略**：
     - 静态物体：Text-to-Image（GPT Image 1.5, $1024\times 1024$ 纯色背景）→ 前景分割（SAM3）→ 单图三维重建（SAM3D）→ VLM 规范化朝向与目标尺寸缩放。
     - 关节物体：从 ArtVIP 数据集基于 8 视角 ViT-H-14 CLIP 嵌入（1024 维）与目标包围盒 L1 距离做两阶段检索（作者弃用了网格和关节质量较差的 PartNet-Mobility）。
     - 薄覆盖物（Thin Coverings，如地毯、桌布、海报）：支持矩形/圆形与平铺/单图两种模式，优先从 ambientCG 检索 PBR 材质，失败则回退到 GPT Image 1.5 生成并做边缘线性混合。
   - **物理与碰撞增强**：VLM 依据 6 视角渲染预测 19 类主导材质（映射摩擦系数）、质量区间 $[m_{\min}, m_{\max}]$（支持域随机化）及功能正面朝向；按均匀密度 $\rho = m / V_{\text{mesh}}$ 计算惯量张量 $\boldsymbol{I}_B = \rho \cdot \boldsymbol{I}_{\text{unit}}$；关节物体采用先整体后分连杆的质量约束估计。碰撞几何采用 V-HACD（家具上限 128 凸块、墙面与小物 64、天花板 16），文中对比指出 CoACD 会导致堆叠物体出现膨胀漂浮效应且计算慢 1–2 个数量级。所有资产需通过 VLM 校验（类型、风格、单物体、完整性、比例、关闭状态），失败则按重试预算重新生成或检索。

4. **专门工具与物理可行性后处理**
   - **布局生成**：`generate_room_specs` 使用带房间排序（anchor → connector，面积降序）、边贴合 11 点采样与紧凑度/稳定性评分的 best-first 回溯搜索，并通过 BFS 检查外门可达性。
   - **家具与小物工具**：`check_facing`、`snap_to_object`（三阶段碰撞消解-朝向对齐-贴合消隙）、基于 Minkowski 腐蚀/膨胀的机器人可达性检查 `check_reachability`；小物阶段提供 Drake 仿真驱动的复合工具 `create_stack`、`fill_container`、`create_arrangement`（全有或全无检验）、`create_pile`；支持按 Natural/Perfect 风格注入高斯位置/偏航角噪声（Table 9）以消除“整米/整角度”机械感。
   - **两步物理后处理**：先在 Drake 中使用 SNOPT 求解非线性非穿透投影优化 $\min_{\mathbf{p}}\|\mathbf{p}-\mathbf{p}_0\|_2^2 \text{ s.t. } d(i,j)\ge 10^{-5}\text{m}$（固定朝向），再运行重力沉降（1ms 步长，5s 模拟），并自动移除倾倒家具与掉落/穿地小物。场景原生以 SDFormat 保存，可导出 MJCF、USD 与 Blender，兼容 Drake、MuJoCo、PyBullet、Isaac Sim 与 Genesis。

5. **实验评测与机器人应用（核心核对数据）**
   - **评测集与基线**：210 条 Prompt（179 房间级：SceneEval-100 房间 94 + Type Diversity 50 + Object Density 25 + Themed 10；31 房屋级：SceneEval-100 房屋 6 + House-Level 25），对比 HSM、Holodeck、SceneWeaver、I-Design、LayoutVLM（Curated/Objaverse）及 6 个消融（NoCritic、NotGenerated、NoAssetValidation、NoSpecializedTools、NoObserveScene、NoAgentMemory）。
   - **人类评测（205 名 Prolific 参与者，3,051 条有效成对比较）**：房间级平均真实感胜率 92.2%（对单个基线 88.5%–95.4%，全部 $p<0.001$）、提示词忠实度胜率 91.5%（85.2%–95.8%）；房屋级对 Holodeck 真实感胜率 80.3%、忠实度胜率 84.7%。消融中 `NotGenerated`（63.8%/67.0%）、`NoAssetValidation`（63.0%/62.2%）与 `NoObserveScene`（61.5% 真实感）降幅显著。
   - **自动化与物理指标（Table 2 / 14 / 16 / 17）**：房间级平均物体数 71.1 ± 13.0（基线 11.2–23.0），CNT 82.9%、ATR 74.4%、OOR 67.6%、OAR 80.6%、ACC 83.4%、NAV 97.6%、OOB 0.2%；物理碰撞率 COL 1.2% ± 0.6%（基线 3.0%–28.9%），平均穿透深度 MPD 3.77 ± 1.28 mm（基线 10.87–47.05 mm），静态平衡率 STB 95.6% ± 1.7%（基线 8.1%–60.8%），平均沉降位移 MD 12.3 ± 4.5 mm（基线 382.3–932.2 mm）。房屋级 214.1 ± 60.9 物体，COL 0.9%、STB 79.8%（Holodeck 为 81.3 物体、COL 3.8%、STB 17.9%）。Base-model swap 对照实验（将 HSM 与 Holodeck 换用 GPT-5.2，Table 15）表明性能优势并非仅来自底座模型升级。
   - **成本与耗时（Table 18 / 19）**：25 场景并发于 8×L40S GPU 时，完整版单房间均价 $13.98（Designer 占 75.9%、Critic 占 20.0%）、均时 3 小时 26 分；`NoCritic` 成本降至 $4.24（降 70%）、耗时 1 小时 40 分，但物体数降至 54.0。
   - **机器人评测闭环与真机风格策略验证**：在 4 个抓取放置任务 × 25 生成场景（共 100 场景）中，对比标准模型策略（VLM 解析 + 点云逆抓采样 + IK + RRT-Connect + TOPP）与刻意退化策略，分别取得 16/100（16%）与 12/100（12%）成功率；Agentic Evaluator（状态查询 + 渲染检查）在 300 个场景状态上与人工标注一致率达 299/300（99.7%）。另在 Drake 中演示了 Rainbow RB-Y1 人形机器人遥操作开关柜及外部预训练 LBM 策略（Lin et al., 2026）零样本推理。

6. **局限与失败模式（Appendix S）**
   - ArtVIP 检索库有限导致少数误检索（如用储物柜替代迷你冰箱），且需放宽校验标准；SAM3D 缺少 PBR 材质通道导致透明/金属物体外观失真。
   - VLM 空间推理在规范化朝向或对称摆设（如餐桌两侧餐具镜像方向）上偶发错误。
   - 顺序阶段构造存在阶段边界泄露（家具生成阶段提前“烘焙”了电视机，而非留给小物阶段），且小尺寸支撑面可能漏检。
   - 不支持可变形体、流体、易碎物，物理参数为自动估计而非实测值，尚未开展正式的 Sim2Real 迁移定量评测。
