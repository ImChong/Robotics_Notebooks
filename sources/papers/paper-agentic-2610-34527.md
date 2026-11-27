# LLM4Robot-Bench: How Far Can GPT-6 Astra Go in Robotic Manipulation? A Hierarchical, Paired Sim–Real Benchmark

- **作者：** Haibao Yu*, Jiaqi Meng*, Zijian Zhang*, Bowen Jing, Lei Yang, Yue Hu, Chenchen Ge, Yiming Hou, Chenbo Xin, Dawen Ye, Haifang Qin, Jin Li, Weitao Zhou, Jianing Qiu, Jiankai Sun, Dandan Zhang, Zaiqing Nie, Ping Luo, Ding Zhao（*共同一作；Haibao Yu 为通讯作者）
- **机构：** Tuojing Intelligence、The University of Hong Kong、Chinese Academy of Sciences、Nanyang Technological University、University of Michigan, Ann Arbor、Southeast University、Sapient Intelligence、SimpleAI、Mohamed bin Zayed University of Artificial Intelligence、Stanford University、Imperial College London、Tsinghua University、Carnegie Mellon University
- **版本 / 载体：** arXiv:2610.34527v3 [cs.RO]（2026-10-25 提交 v1，2026-11-03 修订 v3；31 页，11 张图；v3 将基准名称由 PhysLadder 更名为 LLM4Robot-Bench）
- **原文：** [arXiv HTML 全文 (v3)](https://arxiv.org/html/2610.34527)、[arXiv 摘要页](https://arxiv.org/abs/2610.34527)
- **项目页 / 代码：**
  - 项目网站：<https://llm4robot.github.io/bench/>
  - 当前官方代码库：<https://github.com/LLM4Robot/LLM4Robot-Bench>（Apache-2.0 开源）
  - 历史归档仓库：<https://github.com/PhysLadder-Benchmark/PhysLadder>（2026-11-08 归档，已迁移至 `LLM4Robot/LLM4Robot-Bench`）
  - **开源状态：** 已开源（仿真框架、16 项任务配置与 `variant01` 共 32 组设置、自动评测器及资产/媒体子模块已公开；真机侧通过 Codex Agent Skills 描述接口规范）
- **核对范围：** 已通读 `paper_text.txt` 全部正文与附录（Section 1–7、Table 1–2、Figure 1–10、Appendix A 任务卡与 Table S1、Appendix B 仿真与真机实现细节及 Table S2、Appendix C 基础 Skills 摘要 Figure S1、模型调用次数 Table S3 与瓶颈判定标准 Table S4），并核对 `PhysLadder-Benchmark/PhysLadder` 归档页与 `LLM4Robot/LLM4Robot-Bench` README 及文件清单。
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [LLM4Robot-Bench (PhysLadder) 实体页](../../wiki/entities/paper-agentic-2610-34527.md)

### 1. 研究动机与三层交互难度体系（Section 1, 3.1, Table 1, Table S1）
- 仅统计整体任务成功率无法定位限制基础模型可靠操作的物理交互瓶颈；为此作者围绕**接触配置（contact configuration）**与**力调控（force regulation）**两个耦合维度构建三层累积基准（16 项日常操作任务，覆盖刚体、可变形食品与受约束机构）：
  - **L1（5 项任务：接触选择与几何对齐）**：L1-01 Marker Placement、L1-02 Charger Alignment、L1-03 Hat Hanging、L1-04 Cup Nesting、L1-05 Flower Transfer。要求无妨碍后续操作的稳定抓握与几何对准，不要求超越常规支撑/搬运的额外力调控，也不要求操作中途重构接触。
  - **L2（7 项任务：持续接触下的力调控）**：L2-01 Marker Cap Removal、L2-02 Book Pulling、L2-03 Marble Pouring、L2-04 Orange Handling、L2-05 Drawer Pull & Return、L2-06 Jelly Handling、L2-07 Whiteboard Wiping。在持续接触中克服阻力、控制倾倒流动、避免压碎可变形物体或施加有效擦拭正压力，但不要求在主操作中松开并重抓。
  - **L3（4 项任务：协调的接触重构与力调控）**：L3-01 Bottle Cap Unscrewing、L3-02 Nut Unscrewing、L3-03 Stacked-Book Pushing & Rotation、L3-04 Block Sweeping。单次接触无法完成全程（如瓶盖/螺母旋转行程超出单次抓握转角、双书推转需变换施力点、分散积木需多次扫掠），必须释放并重新建立接触，同时保持或累积既有物理进展。
- **配置依赖性（Section 3.1, 6.3）**：交互层级绑定具体任务版本（初始状态、目标与过程约束）与当前两指夹爪平台，而非仅由任务名称决定。例如若两块积木聚集在簸箕入口附近，一次局部扫掠即可完成，对应 L2；分散三块积木则需要多次接触重构与力调控，对应 L3。同样，预拧松至仅剩短程旋转的瓶盖也属于 L2。

### 2. 物理与视觉匹配的 Sim–Real 环境（Section 4, Appendix B）
- **真机平台（Section 4.1, Appendix B.1）**：双臂 ARX R5（每臂 6 旋转关节 + 1 自由度耦合两指夹爪），无专用接触力传感器；相机包括 1 台外部固定 Orbbec Gemini Pro（640×480 @ 30 fps）和 2 台腕部 Orbbec Gemini 305（848×480 @ 15 fps），仅向模型提供 RGB 图像（不提供深度图）。夹爪指令为归一化开合度 `[0, 1]`（位置目标而非力控制），运动超限触发保护性急停并记为失败。
- **仿真平台（Section 4.2, Appendix B.2, Table S2）**：基于 NVIDIA Isaac Sim 5.1.0，加载开源 ARX R5 模型与三相机视角；利用多视角视频重建场景并按实物尺度（如记号笔 150 mm）与称重质量（如果冻 26.4 g）校准。默认物理步频 240 Hz，拔笔帽/拧瓶盖/拧螺母为 1920 Hz，果冻操作为 360 Hz；采用 TGS 求解器（32 位置迭代，L1 4 速度迭代，其余 0）。橙瓣与果冻采用 GPU 有限元（FEM）可变形模型（果冻参数：$E=3\text{ kPa}, \nu=0.45, \rho=1030\text{ kg/m}^3, \mu_d=0.6$）；白板擦拭采用基于法向力 $F>0.1\text{ N}$ 与滑动死区（0.5 mm–20 mm）的墨迹单元覆盖度阈值（2.5）代理模型。仿真图像渲染为外部 1280×954 与腕部 1280×720，送入模型前长边缩放到 768 像素。

### 3. 模型接口、闭环执行与评测协议（Section 5, Appendix C）
- **闭环决策（Section 5.1–5.2）**：评估 GPT-6 Astra（`reasoning effort = xhigh`，经由 OpenAI Codex CLI 接入），采用任务级 zero-shot 设定（无任务示例或先验轨迹）。输入包括固定任务指令与任务无关接口说明（仿真中由 policy adapter 作为模型指令提供；真机中通过固定 `SKILL.md` Basic Skills 提供）。每步观测为外视角 + 左右腕部 RGB、本体关节/夹爪状态、执行反馈与回合历史。
  - **仿真执行**：每次模型调用返回 1 个结构化动作（目标末端位姿与 0–70 mm 夹爪开合度），由 cuRobo 生成无碰撞关节轨迹并交由 Isaac Sim 关节驱动跟踪。默认单回合预算上限为 150 次模型调用或 5 小时挂钟时间。
  - **真机执行**：模型通过工具调用驱动控制服务与 ARX R5 SDK，按有界分段（bounded segments）移动，无预设任务时间或调用次数硬上限，由模型自报完成/不可行、算子人工停止/接管或触发保护急停结束；最终成功由人工对照执行视频与 Appendix A 判据判定。

### 4. 核心实验结果与瓶颈分析（Section 6, Table 2, Table S3–S4）
- **跨层级成功率（每任务每环境 3 次 trial，等权平均）**：
  - 真机：L1 80.0% (12/15) → L2 47.6% (10/21) → L3 0.0% (0/12)，总平均 45.8% (22/48)。
  - 仿真：L1 66.7% (10/15) → L2 47.6% (10/21) → L3 0.0% (0/12)，总平均 41.7% (20/48)。
- **Sim–Real 一致性**：16 项任务中有 10 项在仿真与真机上取得完全相同的成功次数（3 项全成 3/3：L1-01 Marker Placement、L1-05 Flower Transfer、L2-03 Marble Pouring；7 项全败 0/3：L1-03 Hat Hanging、L2-01 Marker Cap Removal、L2-06 Jelly Handling 及全部 4 项 L3 任务），其余 6 项仅相差 1 次成功（3 次试验中差 1 次）。差异来源包括判据粒度（L1-02 Charger Alignment 仿真采用固定几何容差而耗尽 150 次调用预算 2/3，真机目视确认 3/3）及释放后稳定性（L1-04 Cup Nesting 仿真出现 1 次释放后倾倒 2/3，真机 3/3）。
- **四大交互瓶颈（Section 6.4, Table S4）**：
  1. **空间推理与执行精度**：如 L1-03 Hat Hanging 能抓起帽子却无法把帽口套准挂钩分叉（Sim 0/3, Real 0/3）；L1-02 Charger Alignment 接近插座后仍需反复微调。
  2. **接触配置选择与建立**：如 L2-01 Marker Cap Removal 未能沿轴线建立有效拔帽抓握；L2-04 Orange Handling 无法在无损前提下抓起竖放橙瓣（Sim 2/3, Real 3/3）；L3-02 Nut Unscrewing 双臂接触后后续旋转位姿不可行。
  3. **力调控**：如 L2-02 Book Pulling 难以平衡右臂稳压上层书与左臂抽拉中层书的摩擦力（Sim 1/3, Real 0/3）；L2-05 Drawer Pull & Return 局部拉推无法稳定完成往返（Sim 1/3, Real 2/3）；L2-07 Whiteboard Wiping 擦拭后留有可见字迹（Sim 3/3, Real 2/3）；L3-01 Bottle Cap Unscrewing 旋转中打滑/掉落瓶盖。
  4. **单次与跨步交互协调**：如 L2-06 Jelly Handling 抓握位置与夹紧力耦合导致无法稳固提起果冻（Sim 0/3, Real 0/3）；L3-03 Stacked-Book Pushing & Rotation 仅能部分平移与旋转而无法完成整圈；L3-04 Block Sweeping 难以在多次清扫间协同路径规划、工具握姿与扫掠施力。
- **局限性**：仅评估了 GPT-6 Astra 单个模型且每任务仅 3 次 trial（受真机资源与并行度限制），跨模型 Sim–Real 预测性尚待验证；真机未配备专有力传感器且部分判定（如对齐、擦净）含人工目视主观性。
