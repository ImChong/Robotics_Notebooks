# How to start a conversation proactively?（arXiv:2611.09906）

> 来源归档（paper · 原文通读）

- **标题：** How to start a conversation proactively?
- **类型：** paper
- **作者 / 机构：** Mohit Nathani, Aaditya Gupta, Eshwar Dhande, Chayan Sarkar；TCS Research, India（论文首页）
- **arXiv：** <https://arxiv.org/abs/2611.09906>（v1，cs.HC，首页标注 22 Aug 2026）
- **PDF：** <https://arxiv.org/pdf/2611.09906v1>
- **代码 / 数据：** https://github.com/mohit-nathani1/conversation-opener-dataset （CCO 数据集仓库；仅静态读 README，未下载数据；README 称对应 RO-MAN '26 论文，该会议信息不在论文正文内）
- **入库日期：** 2026-11-10
- **一句话说明：** 把「用户特质 / 用户状态 / 环境状态 / 主动触发」四类变量组织成结构化语境，构建 2,487 条合成数据集 CCO，用 QLoRA 微调 3B~4B 小语言模型生成单句主动开场白；评测只有文本指标与 LLM 裁判，无真人与真机实验。
- **核对范围：** 通读 arXiv v1 PDF 全文：摘要、§I 引言、§II 相关工作、§III 语境 schema、§IV 数据集（表 I、图 3）、§V 微调、§VI 评测（表 II–IV）、§VII 讨论与局限、参考文献列表；该版本无附录 / 补充材料，故无此部分可读。图 1–3 仅读图注与图内文字。未读其引用文献原文；GitHub 仓库只读了 README，未检查 `cco_dataset.json` 内容与许可证，未复现任何数字。
- **通读：** 已完成

## 核心摘录（均出自论文，括号为章节 / 表号）

1. **任务（§IV-C）：** 条件生成 f(trigger, context) → opener，输出为单句、单轮的开场白。
2. **schema（§III，图 2）：** 四部分——user traits（偏好语气、问候形式、啰嗦程度、主动程度等）、user states（活动、情绪、健康状态等）、environmental states（时间、地点、天气、公共事件）、proactive triggers（用药提醒、补水、运动、体检、环境紧急、健康紧急、小聊天）。变量按五种功能角色归类：modifiers、context、constraints、prerequisites、goal states。触发器共用「前置检查 / 语境约束 / 目标状态核验」逻辑。
3. **数据构造（§IV-A，图 3）：** 采样变量组合（含典型与边缘配置）→ GPT-5-mini 生成候选开场白 → LLM 裁判按 6 维 0–5 分打分；均分 ≥ 4.0 直接进 Level-2 验证，否则由 Level-1 标注者迭代改写（最多 3 轮）；三位标注者均为共同作者，两人负责改写、一人负责终验。
4. **规模（§IV-B，表 I）：** 共 2,487 条，train/val/test = 1,760 / 245 / 482（70/10/20）；七类触发。全为合成数据，无真实用户信息（§IV-D）。
5. **微调（§V）：** SFT（下一词交叉熵），QLoRA（4 bit），LoRA 作用于 key projection 层；模型为 llama-3.2-3b-instruct、gemma-3-4b-it、phi-3.5-mini-instruct；单卡 RTX 4090（原文写 16 GB VRAM），按验证集 loss 早停，三模型同一优化设置；输入为指令加 key-value 的纯文本线性化。论文未给学习率、rank、epoch、batch size 等超参。
6. **自动指标（§VI-A/B，表 II）：** 微调 gemma-3-4b-it 在 BLEU-1 0.2864、ROUGE-L 0.3174、BERTScore-F1 0.8962、BLEURT -0.4606、COMET 0.6534、PPL 59.06 上为表中最优（PPL 越低越好，表中 59.06 为最低）；预训练 grok-4-1-fast 对应 BLEU-1 0.1689、ROUGE-L 0.2226。文中称其「所有关键指标最高」。
7. **LLM 裁判（§VI-C/D，表 III）：** 三位裁判（GPT-5-mini、GPT-5、Grok-4.3），好开场白 = 6 维均分 ≥ 4.0；三裁判平均「好」占比：微调 gemma 83.61、微调 llama-3.2-3b 83.40、微调 phi-3.5 77.45；预训练 grok-4-1-fast 70.44、gemini-2.5-flash 66.87、llama-70b 44.99；预训练 SLM 31.95~53.80。
8. **成对比较（§VI-E/F，表 IV）：** 仅 grok-4-1-fast 对微调 gemma，裁判为 GPT-5-mini，482 个测试语境：gemma 胜 337、grok 胜 136、平 9；均分 4.5539 对 4.1169；称配对检验 p < 0.001（未说明具体检验方法）。
9. **作者自述局限（§VII-B）：** 仅单轮开场；微调后是否损害多轮对话未知；依赖人工设计的结构化变量；真实部署需从传感器实时提取并落地这些变量，可靠性与时延待解。

## 与本站说法的核对备注

- 小站「评测以文本指标和模型裁判为主，尚不能代表长期真人互动效果」与论文一致：论文无真人研究，也无机器人实机实验（全文 §VI 只有表 II–IV）。
- 论文引言称对社交机器人重要，但本文实质是 NLP / HRI 文本生成工作，不含强化学习、感知或控制。

## 对 wiki 的映射

- [paper-proactive-conversation-opener-cco](../../wiki/entities/paper-proactive-conversation-opener-cco.md)
