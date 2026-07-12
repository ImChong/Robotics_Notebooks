# Andrej Karpathy 个人站点 — karpathy.ai

- **类型**：个人站点 / 履历、教学、博客与项目索引（原始资料归档）
- **收录日期**：2026-06-06
- **主链接**：<https://karpathy.ai/>
- **抓取说明**：以 **2026-06-06** 对首页公开 HTML 为准；站点自述为纯 HTML+CSS 静态页，无前端框架。

## 一句话

AI 研究者与教育者：OpenAI 创始成员、Tesla AI 总监（Autopilot 视觉与 Optimus 早期）、Stanford CS231n 创始讲师；现以 YouTube **Neural Networks: Zero to Hero** 与 LLM 科普为主线，个人页集中索引职业时间线、代表演讲、教学资源、博客与 pet projects（micrograd、char-rnn、arxiv-sanity 等）。

## 为什么值得保留

- **机器人交叉履历可追溯**：MSc（UBC）做物理仿真 figures 控制器学习；Tesla 时期统领 **车内数据标注 + NN 训练 + 定制推理芯片部署** 全链路；公开演讲含 Tesla AI Day、CVPR FSD、Pieter Abbeel *Robot Brains* 等，便于与本站 [Sim2Real](../../wiki/concepts/sim2real.md)、[Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)、[Locomotion](../../wiki/tasks/locomotion.md) 主题互链。
- **教育主线高信号**：CS231n、Zero to Hero（micrograd / makemore / nanoGPT 等）是视觉与深度学习入门的事实标准入口之一；与本站 [Deep Learning Foundations](../../wiki/concepts/deep-learning-foundations.md) 直接相交。
- **方法论来源人物索引**：其 [LLM Wiki Gist](../blogs/karpathy_llm_wiki_gist.md) 是本仓库 ingest/query/lint 范式的思想来源；个人页提供 **作者级背景**，避免 wiki 方法论页缺少「谁提出、还做过什么」的上下文。

## 履历时间线（来自主页，2026-06-06 抓取）

| 时段 | 角色 / 机构 | 要点 |
|------|-------------|------|
| 2024– | 独立教育者 | YouTube AI 教程；技术轨 Zero to Hero；大众轨 LLM 科普（*Deep Dive into LLMs*、*How I use LLMs* 等） |
| 2023–2024 | OpenAI | 组建 midtraining 与 synthetic data generation 团队 |
| 2017–2022 | Tesla Director of AI | 统领 Autopilot **计算机视觉**：车内标注、训练、定制芯片部署； briefly Optimus |
| 2015–2017 | OpenAI | 研究科学家、创始成员 |
| 2011–2015 | Stanford PhD | Fei-Fei Li 指导；CNN/RNN 与 vision–language；**CS231n 创始设计与主讲师**（2015 150 人 → 2017 750 人） |
| 2009–2011 | UBC MSc | Michiel van de Panne；**物理仿真 figures 的学习控制器**（敏捷机器人 ML 早期） |
| 2005–2009 | Toronto BSc | CS + 物理 + 数学；Geoff Hinton 深度学习课与 reading group |

## 与机器人 / 学习相关的公开材料（主页索引子集）

### 演讲与访谈（robotics-relevant 示例）

| 主题 | 入口 |
|------|------|
| Tesla AI Day 2021 | 主页 *featured talks* |
| Robot Brains（Pieter Abbeel）2021 | 主页链接 |
| AI for Full Self-Driving @ CVPR / ScaledML | 主页链接 |
| Tesla Autonomy Day 2019 | 主页链接 |
| State of GPT @ Microsoft Build 2023 | 主页链接 |

### 教学

- **YouTube — Neural Networks: Zero to Hero**（[播放列表](../courses/karpathy_zero_to_hero_youtube.md)：micrograd → makemore → GPT → GPT-2；配套 [nn-zero-to-hero](../repos/nn-zero-to-hero.md)）
- **Stanford CS231n**（2016 录像、syllabus、notes；r/cs231n）

### 代表写作（主页 *featured writing* 摘录）

| 日期 | 标题 | 备注 |
|------|------|------|
| 2017-11 | Software 2.0 | 学习即编程范式；与策略学习 / 端到端栈讨论相关 |
| 2019-04 | A Recipe for Training Neural Networks | 工程调参 checklist |
| 2016-05 | A Survival Guide to a PhD | 研究路径 |
| 2015-05 | The Unreasonable Effectiveness of Recurrent Neural Networks | 与 char-rnn 同源 |

### Pet projects（主页；GitHub 为准）

- **micrograd** — 极简 autograd + 小型 NN 库（PyTorch-like API）
- **char-rnn** — 字符级 RNN/LSTM 语言模型（Torch）
- **ConvNetJS** — 浏览器内 CNN 训练 demo
- **arxiv-sanity** — arXiv 论文发现与推荐

### 早期机器人相关出版物（主页 *publications* 摘录）

| 工作 |  venue | 与机器人关系 |
|------|--------|--------------|
| Locomotion Skills for Simulated Quadrupeds | SIGGRAPH 2011 | 仿真四足 locomotion |
| Curriculum Learning for Motor Skills | AI 2012 | 运动技能课程学习 |
| Object Discovery in 3D scenes via Shape Analysis | ICRA 2013 | 三维场景物体发现 |

## 对 wiki 的映射

- 升格页面：[wiki/entities/andrej-karpathy.md](../../wiki/entities/andrej-karpathy.md)
- 课程归档：[karpathy_zero_to_hero_youtube.md](../courses/karpathy_zero_to_hero_youtube.md) · [nn-zero-to-hero.md](../repos/nn-zero-to-hero.md)
- 方法论姊妹资料：[karpathy_llm_wiki_gist.md](../blogs/karpathy_llm_wiki_gist.md) → [wiki/references/llm-wiki-karpathy.md](../../wiki/references/llm-wiki-karpathy.md)

## 参考链接

- 个人主页：<https://karpathy.ai/>
- GitHub：<https://github.com/karpathy>
- YouTube：<https://www.youtube.com/@AndrejKarpathy>
- Google Scholar：主页底部链接
