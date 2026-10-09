# The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement

> 来源归档（ingest）

- **类型：** paper / conceptual framework / recursive-self-improvement / ai-agents / governance
- **arXiv：** <https://arxiv.org/abs/2609.11873>（v3，2026-09-22；PDF：<https://arxiv.org/pdf/2609.11873>）
- **作者：** Yi Duan、Ying Liu、Zirui Tang、Haodong Chen、Jun Zhou、Yumou Liu、Bangrui Xu、Yukai Wu、Sidi Chen、Yuhan Zhou、Haoyu Wang、Xiaoyou Yu、Shaokun Han、Xuzhou Zhu、Le Zhou、Bolin Lu、Wei Zhou、Jiachen Liu、Nuozhou Fang、Jiaxin Tian、Ruoyu Chen、Yuxuan Li、Kai Zuo、Kaiyan Zhang、Qianyu Yang、Zijie Wang、Jiantao Qiu、Conghui He、Guoliang Li、Bowen Zhou、Zhiyuan Liu、Zhoufutu Wen、Jihua Kang、Xuanhe Zhou、Fan Wu
- **作者机构：** 上海交通大学、Theseus Labs、清华大学、字节跳动、面壁智能、小红书超级智能团队、上海人工智能实验室、Humanlaya、Agent-Native Research Lab、Frontis.AI
- **项目页：** [Theseus Lab RSI](https://theseus-labs-rsi.github.io/)（另见 [项目页归档](../sites/last-ai-built-by-humans-rsi.md)）
- **配套资源：** [官方 Awesome RSI 清单](../repos/theseus-labs-awesome-rsi.md)
- **入库日期：** 2026-10-09
- **一句话说明：** 提出 Headroom-Closed Index（HCI）与五级自改进自治框架，用于区分“AI参与改进”与能持续改进自身改进过程的递归自我改进；它是概念框架与证据综述，并非新的训练算法或可运行系统。

## 开源状态（项目页核查）

- **项目页：** <https://theseus-labs-rsi.github.io/>，链接至论文及官方 [awesome-rsi](https://github.com/theseus-labs-rsi/awesome-rsi)。
- **代码/数据：** 截至 2026-10-09，项目页及论文未链接可运行的训练、推理或部署实现，也未列出论文专属数据集。关联 GitHub 仓库是 RSI 论文与资源策展清单，仓库声明 CC0-1.0；它不等于本文方法实现。
- **结论：** 论文框架本身未见可运行代码发布；可复用的开放材料是文献索引。该结论按项目页链接核查，不把“有 GitHub 链接”误写成论文已开源。

## 核心摘录（面向 wiki 编译）

### 1. RSI 的判定不应只看“有没有自改进”

论文把真正的递归自我改进视为持续闭环：系统根据经验与反馈改进能力，并进一步改进未来的改进过程。作者以“最后一个由人类建造的 AI”提出问题，但这是一种研究议程/情景表述，不是已到达 RSI 的实证宣告。论文评估当前系统在不同自治环节上的进展，并指出完整递归闭环仍未被证明。

### 2. 五级自治框架

作者把改进自治拆为递进的五层：**改进执行**（执行给定改进方案）、**改进策略**（选择如何优化）、**经验获取**（主动产生有价值的训练/反馈经验）、**环境适应**（面向变化环境调整改进路径），以及**递归元改进**（系统改进自身的改进机制）。前几层出现局部能力，不意味着最后一层闭环已经成立。

### 3. Headroom-Closed Index（HCI）

HCI 试图把异质基准的能力映射为“相对基准进入年份前沿还剩多少提升空间”，按领域标准化后再汇总。论文以领域内进入年份第 90 百分位作为 HCI=0 的参照，以满分为 100，并用样本规模加权的平方根式聚合。它不是跨基准原始分数的直接平均：不同基准必须先做口径归一，证据可靠性也需区分；作者对一手产业自报结果采用折扣。

论文给出的 2026 示例值包括：高等数学 HCI 86.4、研究生科学 85.8、广泛知识 77.2、法律推理 64.5、多模态推理 62.2、前沿学术广度 60.4；交互式软件工程 52.6、搜索/终端 agent 56.8、工具 agent 39.9。它们是作者汇编的跨基准估计，不是单一受控实验的分数，不能脱离测量口径解读。

### 4. 证据成熟度与失败模式

论文覆盖科学研究、具身智能、软件工程与医疗等领域，并讨论学术论文、工业报告、博客、代码仓库与模型文档等异质证据。它给出的数值和案例应区分同行评审结果、公开实验、预印本与厂商自报。作者指出，改进过程的连续性、自动经验采集、环境适应和元改进证据尤其关键；验证器若能被系统利用，也可能引发挑选样本、数据泄漏或评估失真。

### 5. 对机器人研究的映射

具身智能被纳入讨论，但要把“自动调参/训练”“策略在新经验上持续更新”与“机器人系统自主改进生成下一轮改进的方法”分开。机器人硬件上的安全、重置、数据采集、真实环境变化与独立验证仍是闭环成本；论文没有发布可复现的机器人运行时或控制栈。

## 对 wiki 的映射

- **论文实体页：** [paper-last-ai-built-by-humans-rsi](../../wiki/entities/paper-last-ai-built-by-humans-rsi.md)
- **概念互链：** [递归自我改进](../../wiki/concepts/recursive-self-improvement.md)——将五级自治与 HCI 纳入 RSI 判别框架
- **相关综述：** [RSI Survey（2607.07663）](../../wiki/entities/paper-rsi-survey-2607-07663.md)——机制 taxonomy 与验证层级；与本文的自治成熟度框架互补
- **项目页、资源清单：** [项目页归档](../sites/last-ai-built-by-humans-rsi.md) · [官方 Awesome RSI 清单](../repos/theseus-labs-awesome-rsi.md)

## 参考来源（原始）

- arXiv 摘要与版本记录：<https://arxiv.org/abs/2609.11873>
- 论文 PDF（v3）：<https://arxiv.org/pdf/2609.11873>
- 项目主页：<https://theseus-labs-rsi.github.io/>
- 官方关联资源清单：<https://github.com/theseus-labs-rsi/awesome-rsi>
