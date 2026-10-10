# Aether AI 官网 Blog / News 列表核查

- **类型：** 官方站点索引（博客列表 + 新闻列表）
- **链接：** Blog <https://aetherlabs.ai/blog.html>；News <https://aetherlabs.ai/news.html>
- **机构：** Aether AI（总部 San Diego，CA；创始人 UCSD 助理教授 Biwei Huang / 黄碧薇）。官网未写成立日期；成立于 **2026 年春** 据 San Diego Business Journal（2026-07-06）："Huang launched Aether this spring"，<https://sdbj.com/technology/aether-taking-ai-to-its-next-logical-step/>
- **核查日期：** 2026-10-10
- **方法：** `curl -sSL -A "Mozilla/5.0"` 读取 `blog.html` 与 `news.html`（静态 HTML，服务端直出）后去标签，核对标题、日期、链接；列表页未见分页或「加载更多」。官网左侧索引把博文编为 01–11。
- **用途：** 确认公司页 [Aether AI](../../wiki/entities/aether-ai.md) 覆盖全部官方博文与新闻。
- **同名区分：** arXiv 2503.18945 *Aether: Geometric-Aware Unified World Modeling*（署名 Aether Team，Haoyi Zhu 等，2025-03）是另一组作者的项目，与本公司无关，见 [该页](../../wiki/entities/paper-sa-2503-18945-aether-geometric-aware-unified-world-modeling.md)。

## Blog 列表（2026-10-10 共 11 篇）

链接均相对 `https://aetherlabs.ai/`。

| 官网编号 | 官网日期 | 标题 | 官网标签 | 官网入口 | 本库节点 |
| --- | --- | --- | --- | --- | --- |
| 01 | 2026-05-17 | Causality and the Next AI Paradigm | Causality · World Models | [articles/causality-and-the-next-ai-paradigm.html](https://aetherlabs.ai/articles/causality-and-the-next-ai-paradigm.html) | 公司页「奠基四篇」；归档 [aether_foundations_2026-05](../blogs/aether_foundations_2026-05.md) |
| 02 | 2026-05-17 | Causal Copilot: Toward AI That Discovers Before It Acts | Causal Agents · Discovery | [articles/causal-copilot-toward-ai-that-discovers-before-it-acts.html](https://aetherlabs.ai/articles/causal-copilot-toward-ai-that-discovers-before-it-acts.html) | 公司页「Causal Copilot」；归档同上 |
| 03 | 2026-05-17 | Building the Causal Brain of World Agent | World Agents · Memory | [articles/building-the-causal-brain-of-world-agent.html](https://aetherlabs.ai/articles/building-the-causal-brain-of-world-agent.html) | 公司页「奠基四篇」；归档同上 |
| 04 | 2026-05-17 | Learning Causal World Models: A Closed-Loop Recipe for Exploration, Representation, and Decision Making | Causal World Models | [articles/learning-causal-world-models.html](https://aetherlabs.ai/articles/learning-causal-world-models.html) | 公司页「奠基四篇」；归档同上 |
| 05 | 2026-07-09 | Back to Parsimonious Latents: Task-Centric World Models from Visual Foundations | World Models · Representation | [articles/task-centric-world-models.html](https://aetherlabs.ai/articles/task-centric-world-models.html) | 见下方链接（待补） |
| 06 | 2026-07-16 | The Geometry of Contact: Learning Object Manipulation from Scratch | Reinforcement Learning · Manipulation | [articles/the-geometry-of-contact.html](https://aetherlabs.ai/articles/the-geometry-of-contact.html) | 见下方链接（待补） |
| 07 | 2026-07-27 | CD-LAM: Causal Debiasing Gives World Models Stronger Action Control with 10x Less Post-training | Causal AI · World Models | [articles/cd-lam-causal-debiasing-for-embodied-world-models.html](https://aetherlabs.ai/articles/cd-lam-causal-debiasing-for-embodied-world-models.html) | 见下方链接（待补） |
| 08 | 2026-08-09 | SCAR: Self-Supervised Continuous Action Representation Learning | World Models · Latent Actions | [articles/scar-self-supervised-continuous-action-representation-learning.html](https://aetherlabs.ai/articles/scar-self-supervised-continuous-action-representation-learning.html) | 见下方链接（待补） |
| 09 | 2026-09-15 | RSIAgent: Autonomous Exploration for Recursive Self-Improvement in New Environments | Causal Agents | [articles/rsiagent-autonomous-exploration-for-recursive-self-improvement.html](https://aetherlabs.ai/articles/rsiagent-autonomous-exploration-for-recursive-self-improvement.html) | [RSIAgent](../../wiki/entities/aether-rsiagent.md)；归档 [aether_rsiagent](../blogs/aether_rsiagent.md) |
| 10 | 2026-09-19 | CausalWM: Causal Chain-of-Thought Reasoning for Embodied World Model | Causal World Models | [articles/causalwm-causal-chain-of-thought-reasoning-for-embodied-world-model.html](https://aetherlabs.ai/articles/causalwm-causal-chain-of-thought-reasoning-for-embodied-world-model.html) | [CausalWM](../../wiki/entities/paper-causalwm.md) |
| 11 | 2026-10-08 | CRIS-0: Building a Real-world Autonomous Robotic System with Causality-driven Agent and World Model | Causal AI · Robotics | [articles/real-world-autonomous-robotic-system-with-causality-driven-agent-and-world-model.html](https://aetherlabs.ai/articles/real-world-autonomous-robotic-system-with-causality-driven-agent-and-world-model.html) | [CRIS-0](../../wiki/entities/aether-cris-0.md) |

## News 列表（2026-10-10 共 3 条）

| 官网日期 | 标题 | 类别 | 入口 | 本库节点 |
| --- | --- | --- | --- | --- |
| 2026-06-17 | Aether AI Raises $20 Million Seed Round to Build Causal World Models for the Next Era of AI | Company News · Funding | [news/aether-ai-raises-20m-seed-round.html](https://aetherlabs.ai/news/aether-ai-raises-20m-seed-round.html) | 公司页「2026-06 种子轮」；归档 [aether_seed_round_2026-06](../blogs/aether_seed_round_2026-06.md) |
| 2026-09-19 | Aether AI Introduces CausalWM, Its First Embodied Causal World Model | Research · Causal World Models | 指向 Blog 10（同一 URL） | [CausalWM](../../wiki/entities/paper-causalwm.md) |
| 2026-10-08 | Aether AI Introduces CRIS-0, a Causality-driven Robotic Intelligence System for the Real World | Research · Causal AI · Robotics | 指向 Blog 11（同一 URL） | [CRIS-0](../../wiki/entities/aether-cris-0.md) |

## 列表摘要中的自报数字（未经第三方复现）

- **Blog 05 TC-WM：** 单个线性投影即可从冻结视觉基础模型特征中提取紧凑、任务中心的状态。
- **Blog 06 Geometry of Contact：** Interaction-Weighted Resampling 使真实空气曲棍球机器人成功率从 5/20 提到 12/20。
- **Blog 07 CD-LAM：** 先对潜动作空间去偏，动作跟随误差降低 30% 以上，后训练量少 10 倍（标题自报）。
- **Blog 08 SCAR：** 从视觉转移学习统一潜动作接口，使世界模型跨本体迁移动作结构。
- **Blog 09 RSIAgent：** 不更新参数，OSWorld 2.0 78.98%、Agents' Last Exam 84.82%。
- **Blog 10 CausalWM：** 先预测运动与几何再生成未来视频；TriWorldBench TWB-Score 66.04 排名第一，PAI-Bench robot 域第一、超过 Cosmos 3 Super。
- **Blog 11 CRIS-0：** Causality-guided Robot Agent + Causal World Model；追踪任务状态并逐阶段验证，从扰动中恢复、无人干预完成长程任务、按用户调整行为。

## 列表外的相关入口

- **主页 Manifesto**（<https://aetherlabs.ai/>）：因果世界模型定义、Physical AI 为首个验证场、延伸到科学发现（生物 / 医学 / 长寿）。主页 HTML 中「创始团队」与「Scientific Advisors」两块被 HTML 注释隐藏，**页面不展示**，本库不把其中名单当作公开事实。
- **Causal-Copilot 代码 / Demo：** <https://github.com/Lancelot39/Causal-Copilot>（MIT），<https://causalcopilot.com>；见 Blog 02 文末链接。
- **社交账号：** X <https://x.com/AetherLab_AI>；LinkedIn <https://www.linkedin.com/company/116017024>。不计入博文。
- **press 中给出的论文编号（arXiv API 已核对标题）：** TC-WM [2605.25620](https://arxiv.org/abs/2605.25620)；Geometry of Contact 对应 *Learning Object Manipulation from Scratch via Contrastive Interaction* [2606.11525](https://arxiv.org/abs/2606.11525)；SCAR [2605.16412](https://arxiv.org/abs/2605.16412)。来源：机器之心（36 氪转载，2026-06-24）<https://www.36kr.com/p/3866596553561095>。

## 日期口径

本表日期取官网列表页显示日期。融资公告官网列表为 06-17，正文电头为 "June 16, 2026"，GlobeNewswire 通稿为 06-18；公司页按官网列表日期 06-17 记。
