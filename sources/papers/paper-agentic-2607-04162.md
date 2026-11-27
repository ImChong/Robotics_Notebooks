# ACE: Agentic Control for Embodied Manipulation via Zero-shot Workflow Reasoning

> 原始论文资料归档；主身份 `arXiv:2607.04162`（v1 标题为 *ACE: Agentic Control for Embodied Manipulation via Zero-shot Workflow Reasoning*，v2 HTML 正文标题更新为 *ACE: Semantic Task Composition for Tabletop Manipulation via Tracked Pick-and-Place Masks*）。

- **标题：** ACE: Agentic Control for Embodied Manipulation via Zero-shot Workflow Reasoning（arXiv v2 正文标题：ACE: Semantic Task Composition for Tabletop Manipulation via Tracked Pick-and-Place Masks）
- **作者：** Iok Tong Lei*、QianZhi Li*、Ying Jie Yap、Yujie Zhang、Rui Zhong、Haichao Gui†、Xiaolong Liu†、Zhidong Deng†（* 共同一作，† 通信作者）
- **机构：** 清华大学计算机科学与技术系（Tsinghua University）、北京航空航天大学国家卓越工程师学院/宇航学院（Beihang University）、独立研究者、无锡灵巧手机器人科技有限公司（Wuxi Dexteroushands Robotic Technology Co.）
- **版本 / 载体：** arXiv:2607.04162v2 [cs.RO]（v1 提交于 2026-07-05，v2 修订于 2026-09-30，Preprint）
- **原文：** [arXiv 摘要页](https://arxiv.org/abs/2607.04162) · [arXiv v2 HTML 全文](https://arxiv.org/html/2607.04162)
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27，arXiv 摘要、v2 HTML 全文与项目索引均未列出 ACE 官方项目页或 GitHub 代码仓库链接，且未见开源发布承诺；系统底层依赖外部公开组件 Qwen-Agent、Qwen3.8-27B、SAM 3、Cutie 与 LeRobot Diffusion Policy）。
- **核对范围：** 通读 arXiv:2607.04162v2 HTML 全文 §I–§VIII（Introduction、Related Work、Method §III-A–E、Experiments §IV-A–D、Results §V、Ablations §VI、Limitations §VII、Conclusion §VIII）、Appendix A（任务形式化定义）、Appendix B（实现配置、FGS 评分规则 B-A、消融对照设定 B-B）、Table I 与 Table II 完整实验数据及 References [1]–[29]。
- **通读：** 已完成

## 原文核对笔记

### 1. 核心问题与系统定位（§I–§III）

- **问题切口：** 通用桌面操作既需要针对开放指令的语义约束推理，又需要接触密集动作的稳定执行。直接用端到端策略在长程语义组合任务上学习容易在少样本（约 1 小时演示）下失效。
- **Task-level zero-shot composition：** 底层 Vision-Action（VA）策略 $\pi_{\mathrm{VA}}$ 仅在约 1 小时的通用掩码条件抓取-放置演示 $\mathcal{D}_{\text{prim}}$ 上训练，**不含**两个评测任务 $\mathcal{T}_{\text{eval}}$ 的完整任务轨迹；推理时由高层智能体完成对象选择、角色-目标绑定与步骤编排。
- **两大机器人技能接口（Qwen-Agent + Qwen3.8-27B）：**
  1. `MaskInterface`：给定工作流步 $w_k = (o_k, d_k)$、当前帧 $I_t$ 与记忆状态 $m_t$，调用 SAM 3 生成统一 8-bit 灰度掩码 $M_k \in \{0, 127, 255\}^{H \times W}$（像素值 `127` 编码抓取目标 `pick target`，`255` 编码放置目标 `place target`，`0` 为背景），并由 Cutie 在后续帧连续追踪更新 $(T_k, M_{k,t+1})$。
  2. `PickPlace`：使用 LeRobot Diffusion Policy（ResNet-18 ImageNet 预训练视觉主干 + 32 spatial-softmax keypoints，输入复制为 $3 \times 480 \times 640$ 的灰度掩码与 6 维本体状态 $q_t$）以滚动时域方式执行动作块 $\hat{a}_{t:t+L-1} = \pi_{\mathrm{VA}}(q_{t-1:t}, M_{k,t-1:t})$，预测步长 64、每次执行 $L=32$ 步。
- **四层记忆分离（§III-D）：**
  - **Live execution memory：** 快时间尺度维护工作流位置、当前掩码、Cutie 追踪状态与执行进度。
  - **Task-scoped semantic-visual memory：** 记录对象名称、别名、角色与视觉关联（含工具 `record_task_object_knowledge`）。
  - **Appearance reference memory：** 保存已知对象的图像特征或裁剪图，用于后续重新定位（re-grounding）。
  - **Conversational context memory：** 有界智能体调用历史（初始指令、推理与工具调用记录）。
- **闭环验证与重试上限（§III-E）：** `Ground -> Execute -> Assess`。若子目标完成则推进；否则重定位并重试。若智能体对**同一激活子目标**连续 3 次判定未完成，则终止当前试验。

### 2. 硬件平台与核心实验结果（§IV–§VI, Tables I–II）

- **实验硬件与算力：** SO-101 机械臂；智能体推理引擎部署在云端 4× NVIDIA RTX 3090 服务器，掩码接口与 VA 策略运行在本地单张 RTX 3090。每组设定均在随机初始布局与同义改写指令下进行 20 次自主真机试验。
- **Table I（主结果，每项 20 次真机试验）：**
  - **Semantic Formula Assembly**（选两个数字块与一个运算符块组成满足目标值的算式，依次放入 A/B/C 纸区；FGS 满分 20）：
    - ACT（约 1 小时全任务演示）：avg. FGS `3.2`，SR `0.0%`
    - $\pi_{0.5}$（约 1 小时全任务演示微调）：avg. FGS `3.0`，SR `0.0%`
    - ACE（w/o persistent memory）：avg. FGS `16.8`，SR `55.0%`，GA `96.7%`
    - ACE（Full Framework）：avg. FGS `18.0`，SR `70.0%`（14/20），GA `100.0%`
  - **Constraint Retrieval**（按逻辑/数值约束筛选目标并放到指定区域，如 $<4$ 放左侧、$>4$ 放右侧）：
    - ACT：SR `0.0%`
    - $\pi_{0.5}$：SR `0.0%`
    - ACE（w/o persistent memory）：SR `70.0%`，GA `90.0%`
    - ACE（Full Framework）：SR `80.0%`（16/20），GA `97.5%`
- **Table II（消融与独立控制器对照，每项 20 次真机试验）：**
  - **(a) 单方块抓取成功率：** $\pi_{0.5}$ LoRA（300 条单方块抓取演示微调）在 Seen 上 `60.0%`、Seen + Unseen 上 `40.0%`；DP（Mask + RGB）在共享测试集上 `30.0%`（6/20）；ACE / DP（Mask Only）在共享测试集上 `90.0%`（18/20），较 Mask + RGB 高出 60 个百分点。
  - **(b) 全任务演示下的独立 DP 对照：** Standalone DP 在 Semantic Formula Assembly 上 avg. FGS `3.4`、SR `0.0%`，Constraint Retrieval 上 SR `0.0%`。
- **关键定性观察（§VI）：** 无持久记忆时，方块在抓取或放置过程中发生旋转会遮挡表面印刷符号，导致后续观测无法辨认原物体，引发“任务是否已完成”的判定歧义；持久记忆保留了早先的对象绑定与任务事实，将两项任务的成功率分别提升 15 与 10 个百分点。

### 3. 局限性与边界（§VII）

- 评测仅覆盖两个方块操作任务和单一的抓取-放置原语，尚未扩展到多样物体几何与非 pick-and-place 技能。
- 持久对象记忆与对话历史记忆在消融中被联合移除（`w/o persistent memory`），未单独拆解各自贡献。
- 恢复行为的观察为定性记录，验证准确率、重试成功率和端到端耗时未作单独量化统计。

对 wiki 的映射：[ACE 论文实体页](../../wiki/entities/paper-agentic-2607-04162.md)
