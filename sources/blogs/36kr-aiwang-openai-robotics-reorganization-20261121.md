# OpenAI为何撤销机器人团队？当最大的发明已经确认，不要幻想出现第二个奇迹

- **资料类型：** blog / 行业评论（非学术论文、非 OpenAI 官方立场）
- **英文对照标题：** Why Did OpenAI Disband Its Stand-Alone Robotics Team? A Commentary on General Models and Embodied AI
- **署名：** AiWang；首发来源「无话不说聊AI」（ID：AiWang51）；36氪转载
- **原文链接：** https://mp.weixin.qq.com/s/KH3U69OgxJnlQUjDd4Hl4w
- **发布时间：** 2026-11-21 09:12（用户提供的转载信息）
- **入库日期：** 2026-11-21
- **版本与取得方式：** 用户在本次对话直接提供文章全文；微信页面在本次在线访问中返回错误，未能与线上页面逐字互校。内容以用户提供的全文为核对对象。
- **核对范围：** 通读所提供的完整正文，从 OpenAI 组织调整、文字与 Transformer 的「累积+偶然」类比、GPT-6 Astra / Claude / Gemini 机器人能力、世界模型投资风险、分层控制与责任认证、认知卸载、三项冒进指标，至全文结论；对主要新闻、基准与产品案例补做外部一手交叉核对。历史哲学、心理学引用和所有行业预测未逐条查证。
- **通读：** 已完成
- **资料身份：** 观点文章，不提供算法训练/推理源码、权重或可运行实验；新闻与论文链接属于其所依据的外部证据，非本文自有成果。
- **知识映射：** [OpenAI 机器人团队调整与通用模型路线：观点和证据审计](../../wiki/entities/openai-robotics-reorganization-aiwang-2026.md)

## 文章主张与论证链

1. **事实起点：** 将 OpenAI 撤销独立机器人团队视为「通用模型能力进入具身应用」的组织信号，而不是彻底放弃机器人。
2. **历史类比：** 作者认为文字与 Transformer 都来自长期前置条件积累叠加偶然实现，均具认知外化效应；进一步提出「文明级发明不应期待短时间内复制」的投资性结论。
3. **技术论据：** 引 NVIDIA Astranomy RoboLab-120 上 GPT-6 Astra Direct 66.3% / Agentic 74.8%、Anthropic Claude Fable 5.2 RoboDojo Sim 64.9%、Google Gemini Robotics 2.5 工厂装盘案例，论证通用模型逐渐承担部分机器人控制与规划任务。
4. **商业主张：** 作者反对把独立物理世界模型当成「第二个 Transformer」，主张通用模型做理解与规划、专用边缘控制保障实时/安全，创业价值在数据闭环、场景集成与稳定部署。
5. **社会主张：** 认知外包的核心风险是人类放弃验证、判断和责任，优先建设可审计、可追溯的部署制度。

以上是**文章主张的归纳**，不是对「世界模型必然失败」或「通用模型能替代一切 VLA」的背书。

## 关键事实核查（截至 2026-11-21）

| 文中说法 | 交叉核验结果 | 来源与边界 |
|---|---|---|
| OpenAI 撤销独立机器人团队 | **有直接新闻来源支持**。《财富》2026-11-13 报道 OpenAI 向其确认重组，研究人员转入通用研究团队，应用机器人/硬件转至 Applied 部门；并明确机器人仍属长期愿景 | [Fortune 原始采访报道](https://fortune.com/2026/11/13/openai-disbands-robotics-team-as-its-general-models-start-running-robots/)；不是「OpenAI 全部停止机器人研究」 |
| NVIDIA RoboLab-120 Direct 66.3%、Agentic 74.8%、Lakeside 51.0% | **数字存在于 Astranomy 报告**，但 51.0% 仅为该报告列出的对照之一，不能等同于长期固定的「全球最强专用策略」 | [NVIDIA Astranomy](https://research.nvidia.com/labs/srl/projects/astranomy/)；同为仿真，推理时暂停物理过程、每任务 5 次运行 |
| 机器人行业专用模型整体不如通用模型 | **不被上述实验普遍证明**：任务集主要是 RoboLab-120 仿真抓放，未检验长时移动、复杂接触或高动态真机；该报告 Direct 平均推理 18.56 秒/调用 | [NVIDIA 方法、时延和局限](https://research.nvidia.com/labs/srl/projects/astranomy/) |
| 51.0% 是目前榜单最强 | **不成立（不同时间与口径）**：RoboLab 官方实时榜截至本次核对显示 Agentic LM AMP1 86.0%（2579/3000 次），Lakeside 51.0%；榜单含 LM、VLA、WAM 等不同类别及不同 episode 数 | [RoboLab 官方榜单](https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html)；不可把报告图表当实时榜 |
| Claude Fable 5.2 RoboDojo 64.9%、Astra 29.8% | **Anthropic 官方自报**，是在其同一内部评测 harness 下的比较，非与 NVIDIA RoboLab-120 直接可比 | [Anthropic 官方发布](https://www.anthropic.com/claude-fable-and-mythos-5-2) |
| Gemini Robotics 2.5 在奔驰生产线完成装盘 | **Google DeepMind 官方发布有对应概念验证**：Apptronik Apollo 2 工业零件 kitting 试点报 96% 成功率，并用 Robotics ER 管流程、2.5 action model 控制机械手 | [Google DeepMind 官方发布](https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/)；单场景试点不是普适工业部署认证 |
| “文字与 Transformer 是两个几千年级发明”“绝不会有第二个奇迹” | **作者的哲学/投资判断**，没有可验证的统计概率定义，不能从一项组织决定或仿真 benchmark 推导出必然结论 | 文内历史类比与作者论证；建议作为可争议假说阅读 |

## 文章的三个实际价值点

- **模型与控制分工：** 大模型可输出高层决策/任务目标，但实时安全回路要以端侧控制、执行反馈和硬件限幅保障；高层成功率不覆盖时延约束。
- **投资/工程检验：** 数据闭环应报告新增数据对任务覆盖、成功率和成本的边际贡献；产品化指标应包括恢复、接管、合格节拍、可追溯责任。
- **模型“取代”必须可证伪：** 同一任务、同等观测、同样 wall-clock 时间、相近硬件与成本、安全限制下比较 Generalist-only、VLA-only 与 Hybrid，不能只挑单一仿真榜对比。

## 尚未逐一证实的论据

- 文中有关文字起源的因果单线解释、阅读脑区、认知外化「继文字之后第二次」、智力/认知退化等，涉及跨学科争论；本次仅归档作者论点，不提升为已证实定论。
- 商汤「八平台」双臂插花案例、视频时代数据稀缺「高一个量级」、PearsEVO 估值轨迹及作者预测「专用模型只撑一年」未找到并核验完整原始实验报告，不用于定量结论。
- 作者三项「冒进指标」为管理性启发式，文内没有可复现的财务/因果检验协议；不可与技术基准结论同列。

## 参考

- [用户提供的 36氪/AiWang 文章全文及微信原始链接](https://mp.weixin.qq.com/s/KH3U69OgxJnlQUjDd4Hl4w)
- [Fortune：OpenAI stand-alone robotics team reorganization（2026-11-13）](https://fortune.com/2026/11/13/openai-disbands-robotics-team-as-its-general-models-start-running-robots/)
- [NVIDIA：Astranomy](https://research.nvidia.com/labs/srl/projects/astranomy/)
- [RoboLab-120 官方榜单](https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html)
- [Anthropic：Fable 5.2 / Mythos 5.2](https://www.anthropic.com/claude-fable-and-mythos-5-2)
- [Google DeepMind：Gemini Robotics 2.5](https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/)

> 本归档保留作者署名、链接及论点结构，**不转载受版权保护的原文全文**；技术核验和工程建议是本知识库独立编译的注释。