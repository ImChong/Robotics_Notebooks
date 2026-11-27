# DiagGen: Agentic Generation of Deformable Assets with Sim-based Diagnostics for Robotic Simulation

- **作者：** Guanxiong Chen、Yiduo Qu、Qianjun Xia、Pengyu Jing、Yixian Cheng、Bole Ma、Pengzhi Yang、Bingyang Zhou、Ziming Li、Shashwat Suri、Gongbo Sun、Chao Liu、Peter Yichen Chen、Ziqiu Zeng、Fan Shi
- **机构：** 英属哥伦比亚大学（University of British Columbia）、新加坡国立大学（National University of Singapore）、剑桥大学（University of Cambridge）、埃尔朗根-纽伦堡大学（NHR@FAU, Friedrich-Alexander-Universität Erlangen-Nürnberg）、威斯康星大学麦迪逊分校（University of Wisconsin–Madison）
- **版本 / 载体：** arXiv:2609.23103v1 [cs.RO]（2026-09-19 提交）
- **原文：** [arXiv HTML v1](https://arxiv.org/html/2609.23103)、[arXiv 摘要页](https://arxiv.org/abs/2609.23103)
- **项目页 / 代码：** <https://diaggen.github.io/>（项目页，含 3D 资产交互预览、对比视频与引用信息）；<https://github.com/diaggen/diaggen>（**已开源**：公开代码仓 `diaggen/diaggen`，默认分支 `master`，包含面向 Agent 的技能套件 `.agents/skills/run-diaggen-pipeline/`、流水线初始化入口 `hag4r/agentic/run_diaggen_pipeline.py`、Genesis 实时诊断 MCP 服务 `hag4r/agentic/diagnostic_mcp_server.py`、确定性几何/网格/材料工具 `hag4r/tools/` 及配置 `configs/`；仓库元数据未标明开源协议，部分模型权重与 `third_party/` 外部依赖需本地配置）
- **核对范围：** 全文通读 arXiv HTML v1（`paper_text.txt` 第 1–397 行，48,170 字符），覆盖 Abstract、§I–V、Table I–II、Fig. 1–6 题注与正文分析、References [1]–[32]；并核对项目主页快照 `project_page.txt`（第 1–208 行）及官方仓库 `diaggen/diaggen` 的 `README.md` 与 `files_sample` 目录清单。
- **通读：** 已完成

## 原文核对笔记

### 1. 研究动机与定位（§I, §II）

1. **现有可变形资产生成的两大缺口：**
   - **后置校验不回流：** 现有 Agent 资产生成框架（如 PhysX-Omni）通常在生成完成后做一次性或事后物理合理性检查，未能把仿真器中的真实交互物理响应作为诊断信号，定位并修复上游生成阶段的错误。
   - **材料参数脱离部件语义：** 以往基于梯度的材料拟合（如 PhysTwin、EMPM）依赖昂贵的真实交互采集与逐物体优化，未按部件语义分配力学属性，也难以覆盖「柔软本体 + 刚性部件」（如卷发棒机身与柔性电源线、软玩具与金属钥匙圈）的异质对象。
2. **核心思路：** 基于单个非标定日常照片，结合部件级 VLM 语义推理与「物理仿真入环（simulator-in-the-loop）」主动探针诊断，通过 **generate–simulate–diagnose–refine** 闭环将可追溯的仿真证据路由至责任阶段重新生成。

### 2. 六阶段生成与诊断修复架构（§III）

1. **三层系统架构（§III-A, Fig. 2）：**
   - **上层技能代理（Skilled Agents）：** 决策各阶段输出、规划诊断锚点与修复路由；
   - **中层模型后端（Model Backends）：** 支撑 VLM 推理、SAM3 分割与 OmniPart 3D 部件生成；
   - **底层确定性工具（Tools）：** 执行版本化产物契约校验、网格/尺度转换与仿真器接口通信。
2. **六阶段流程：**
   - **图像清理（Image Cleanup）：** 去除背景噪声，生成干净参考图与面向仿真的静止形状/材料描述。
   - **部件分割（Segmentation）：** 使用 SAM3 [27] 将参考图分解为 2D 语义部件掩码。
   - **部件 3D 几何（OmniPart）：** 基于 OmniPart [11] 将掩码与描述提升为部件分离的 3D 网格。
   - **材料推理（Material Inference）：** 由 VLM 代理按部件赋予表面/体填充模式（fill mode）与线弹性物理参数（杨氏模量 $E$、泊松比 $\nu$、密度 $\rho$、摩擦系数、壳层厚度）。
   - **网格处理（Mesh Processing）：** 确立真实物理尺度（metric scale），进行体网格（四面体）剖分，并将分离的部件网格**一体化（monolithization）**为单个连通网格以适配 Genesis 隐式有限元（FEM）仿真。
   - **仿真器诊断与导出（Simulator-Based Diagnostics & Export）：** 在 Genesis 中运行主动抓取探针；通过后冻结纹理与资产包并记录完整谱系。
3. **仿真主动诊断协议（§III-B, Fig. 3）：**
   - 诊断代理根据部件索引、几何摘要与材料规格选择最多 **4 个**行为信息丰富的语义锚点（如手柄、边缘、翻盖、柔性体）并提出物理假设，编译成接触抓取框（grasp box）并先渲染预览微调目标位置。
   - 每次试验从静止重置态出发，执行 **approach–contact–displace–release–settle** 协议，接收顶视/斜视同步 RGB 三联图（triptychs）、部件落地叠加图、几何/动作元数据、步级摘要与运行日志；证据不足时可追加探针。
   - 将异常归因于几何、尺度、分割、表征、材料响应、离散化或测试回合设置，由编排代理（orchestrator）结合修订历史裁决：接收、触发修复或中止。
4. **证据引导的三类上游修复路由（§III-C）：**
   - **Segmentation 路由：** 修复部件合并或碎片化（复用清理后参考图）。
   - **Material inference 路由：** 修复填充模式、壳厚或部件力学参数（复用参考图、掩码与 OmniPart 网格）。
   - **Mesh processing 路由：** 修复度量尺度、剖分精度或连通性故障（同时复用材料推理结果）。
   - **Episode setup：** 留在诊断内部修正抓取目标与预览，不触发上游重建。每次上游修复均重跑选定阶段及其后全部下游阶段，形成可横向对比的新版本。

### 3. 评测指标与实验结论（§III-D, §IV）

1. **静态指标 PMSC（§III-D, Eq. 1）：** 两名独立盲评 VLM 评估源图、分割、几何与材料规格，$S = \max(1, H - F - \mathbf{1}[T > 40{,}000])$，其中 $H \in [1, 5]$，$F \in \{0, 1\}$ 为严重根因扣分，$T$ 为四面体数量（超过 40,000 扣 1 分）。
2. **动态指标 PAIP（§III-D）：** 在修复前冻结基线诊断的单一探针（部件、锚点与动作），在相同仿真器、控制器与相机设置下对比基线与修复版表现；由两名独立 VLM 评委（`GPT-5.6 Terra` Max reasoning，区别于生成与诊断所用的 `GPT-5.6 Luna`）交换 A/B 顺序各评一次（共 4 次判决）汇总胜负平。
3. **注入故障诊断（RQ1, §IV-A, Table I）：** 使用 `GPT-5.5`（high reasoning）对 3 个单阶段注入故障对象（卷发棒电源线赋予刚性模量 → `material_inference`；按摩器金属弹簧掩码残缺 → `segmentation`；厨房夹尺度由 0.28 m 篡改为 2.8 m → `mesh_processing`）做盲测，**3/3** 均准确识别现象并路由至目标上游阶段。
4. **40 个资产诊断与修复评测（RQ2, §IV-B, Table II）：**
   - 40 个来自手机实拍与 MVImgNet2.0 [28] 的资产中（最多允许 1 次修复）：**26 个**首次即通过诊断（平均 PMSC **3.12**）；**14 个**被路由修复。
   - **14 个修复资产 PMSC：** 平均分从 **2.14** 提升至 **3.43**（**+1.29**），其中 **10 个提升（71.4%）**、**4 个持平（28.6%）**、**0 个下降**。
   - **14 个修复资产 PAIP：** 修复版胜 **9 例**、基线胜 **4 例**（如毛绒小猫、运动手表在局部抓取响应上反而退步）、平局 **1 例**。
   - **Table II 典型策略改写：** USB 集线器经 `segmentation` 分离外壳/线缆/金属插头；果汁盒因局部自间隙小于统一壳厚在 Genesis 初始化前被拒，经 `material_inference` 下调壳厚并将单一 3.0 GPa 模量拆分为 0.8/0.7/0.6/0.3 GPa 后通过；医疗包经 `mesh_processing` 提升缺损肩带与五金件网格精度，以 13,318 个四面体恢复全部 5 个部件标签。
5. **下游机器人操作与 Real2Sim 案例（§IV-C, Fig. 4–6）：**
   - 在 Genesis 隐式 FEM + SAP 摩擦接触 [29] 仿真中，利用 `GPT-5.6 Sol` 完成钥匙圈软玩具、厨房夹与饼干、USB 集线器的抓取—搬运—放置轨迹优化。
   - 与未一体化的 OmniPart 四面体网格相比，DiagGen 网格在抓提时保持整体弹性传力而不散架；修复后的 USB 集线器线缆可在重力下自然下垂弯曲。
   - 替换 Agentic Real2Sim [20] 中的 SAM3D 资产与 MuJoCo 后端为 DiagGen 资产与 Genesis 后端，在 Airbot 机械臂与 Meta Quest 2 遥操作采样的真实轨迹上完成数字孪生重建与回放（项目页展示了毛绒玩具抓取放置、恐龙双手扭转与瓶子抓取倾斜对比）。

### 4. 局限性（§V）

1. 材料推理与部件分割故障可能产生相似的外观/物理症状（如过硬行为既可能来自模量过高，也可能来自与刚性外壳粘连），诊断代理在根因判别上仍可能受混淆。
2. 大规模并行化资产生成流程尚未经过测试。
3. PAIP 结果显示部分修复资产（4/14）在交互 Plausibility 上不及基线，且主评测依赖 VLM-as-a-Judge（因缺乏带真机测量标注的部件级可变形材料数据集）。

对 wiki 的映射：[DiagGen 论文实体](../../wiki/entities/paper-agentic-2609-23103.md)。
