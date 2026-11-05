# Proactive AI for Superintelligent Agents

> 来源归档（paper）：203 页分 7 段逐页读完（第 116–203 页为参考文献），主会话对 12 个关键数字回原文抽查一致；要点已并入 wiki 实体页。

- **标题：** Proactive AI for Superintelligent Agents（面向超智能体的主动式 AI）
- **类型：** 综述论文，2026 年 manuscript
- **作者：** Ke Yang、Jiateng Liu、Jize Jiang 等
- **机构：** 伊利诺伊大学厄巴纳-香槟分校、Google、斯坦福大学、加州大学伯克利分校、卡内基梅隆大学
- **论文 PDF：** <https://storage.googleapis.com/gweb-research2023-media/pubtools/1048480.pdf>（Google Research 提供的版本，203 页；未标注版本号）
- **Google Research：** <https://research.google/pubs/proactive-ai-for-superintelligent-agents/>
- **项目页：** <https://empathyang.github.io/Proactive-AI-for-Superintelligent-Agents/>
- **官方仓库：** <https://github.com/EmpathYang/Proactive-AI-for-Superintelligent-Agents>
- **入库日期：** 2026-11-05

## 已核对内容

- 摘要将主动性描述为无需明确请求而发起有目的行动的能力，并提出六项能力：Awareness、Anticipation、Agenda、Arbitration、Action、Adaptation。
- 摘要说明文章综合任务、机制和用户层面的评估，并比较 13 个应用领域的主动性与授权程度；这是一篇综述，不是单一机器人控制算法的实验报告。
- 项目页链接的官方仓库公开了综述论文、参考文献清单及站点相关内容；是否包含可运行的主动式智能体实现尚待检查，不能把网站代码等同于该类实现。

## 论文要点（全文阅读后的摘要）

- **PDF：** Google Research 提供的版本，`pdfinfo` 确认 203 页（未标注版本号）；官方仓库另附一份 `Proactive-AI-for-Superintelligent-Agents.pdf`，**两份是否一致未比对**。
- **定义（§2.1）：** 主动式 AI 智能体指无需针对该干预的新显式请求而发起面向目标行动的自主系统；核心是自我发起（self-initiation）。论文用「委托的剩余裁量（delegated residual discretion）」描述主动性（图 2）。
- **两个问题（§2.2）：** 做什么（what to act on）与是否、何时、如何做（whether/when/how）。
- **光谱（§2.3，图 4）：** 反应式 → 混合主动 → 自我发起；多数系统是混合体，人的角色从操作者 → 协作者 → 评审者。
- **六项能力（§4，图 6）：** 感知（Awareness）→ 预判（Anticipation）→ 议程（Agenda）→ 裁决（Arbitration）→ 行动（Action）→ 适应（Adaptation）；论文称 Anticipation 与 Arbitration 是主动式特有的两个难点。官方仓库另按这六项加 13 个应用领域整理其所引 1,410 篇文献。
- **「校准的主动性」（calibrated proactiveness）：** 把主动程度调到与不确定性、风险、时机、用户偏好与权限相称（§1、§7）。
- **评估（§5）：** 任务级、机制级、用户体验级三层；用户体验含负担、可用性、控制感、信任与依赖（目录）。
- **具身与机器人（§6.4，表 6）：** 论文判断多数已部署具身系统仍是「反应式自主」或「有界主动」，真正开放式主动尚缺可靠性与通用性；列举 Gemini Robotics 2 / ER 2、Helix 2.5、π0.6 (RECAP)、GR00T N1 / N1.7、1X NEO、Waymo Driver、CoFRIDA、ProAct、NIA、RoboGuard 等，并把它们映射到六项能力。评估上指出 ProActBench、RobotEQ 含「该保持沉默」的负样本，NIABench 惩罚打断人的计划，当前视觉语言模型远低于人类且对提示措辞脆弱。
- **跨行业趋势（§6.14）：** 共享架构（基础模型内核 + 领域专用部件）；从辅助人到承载整条工作流；验证与反馈基础设施是关键约束；没有任何领域到达「议程生成式」，多数领域收敛于「受授权约束的使命导向」；具身领域标注为 Reactive–mandate 主动性、Gated action 授权，瓶颈为时延、物理不可逆性与泛化。
- **治理（§7.2）：** 内化主动推理的安全护栏、可信与适当依赖、同意 / 能动性 / 问责边界、道德自我与伦理。7.2.1 主张对内部主动回路加机制级护栏：审计用户模型与议程的构造、防止偏差在推理链中累积、治理「私有上下文 → 主动发起」的转换、并在执行与适应阶段补护栏；7.2.3 提出风险分层的同意模型并要求责任可追溯到同意边界与裁决记录；7.2.4 区分操作性自我监控与拟人化的道德自我主张。

## 开放范围（官方仓库，浅克隆静态阅读）

- 仓库 MIT 许可；内容是综述配套资源：论文 PDF、`data/*.json`（`papers.json` 1,410 篇、`taxonomy`、`evaluation`、`maturity`、`applications`、`site`）、网站（`index.html`、`site/`、`classic/` 视频概览）、`assets/` 图与视频、`scripts/papers.py`。
- **没有可运行的主动式智能体实现**；唯一脚本 `scripts/papers.py` 是文献库校验 / README 生成 / 统计工具（仅标准库）。**本环境拒绝执行仓库代码，该脚本未运行**，时序图基于静态阅读。
- `papers.json` 的 `updated` 为 2026-10-07，1,402 条 `verified`、8 条 `needs-metadata`（我读 JSON 统计）。

## 阅读方式与未核实

- 分 7 段读取：第 1–12 页（引言、概念、历史）、13–24（感知 / 预判 / 议程）、25–46（裁决 / 行动 / 适应 / 评估）、47–69（编程 / GUI / 信息获取 / 具身）、70–96（其余 9 个应用领域）、97–114（跨行业趋势 / 未来工作 / 治理 / 结论）、115–203（参考文献，无附录）。各段由独立读取者逐页读完并写笔记，再并入 wiki 页；我回原文抽查了 12 个关键数字。
- 参考文献按行拆条估计约 1,450–1,500 条（估算，非精确值），与 README 的 1,410 篇数量级吻合。
- **未核实：** 论文引述的各商业 / 研究系统与 2026 年的公司页、论文，按论文口径转述；Google Research 版 PDF 与仓库附带 PDF 是否一致未比对。

## 对 wiki 的映射

- [paper-proactive-ai-superintelligent-agents](../../wiki/entities/paper-proactive-ai-superintelligent-agents.md)
