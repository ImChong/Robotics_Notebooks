# CRIS-0: Building a Real-world Autonomous Robotic System with Causality-driven Agent and World Model（Aether AI）

> 来源归档（blog / Aether AI 官方 + 新闻稿 + 媒体报道）

- **标题：** Building a Real-world Autonomous Robotic System with Causality-driven Agent and World Model
- **类型：** blog（官方技术博文，非 peer-reviewed；无论文 / arXiv）
- **作者 / 组织：** Lingjun Mao†, Lukun He†, Jinglin Cao†, Wenpeng Xu†, Yuchen Yan, Yifei Shao, Junbo Huang, Fang Nan, Ruobin Han, Ziqiao Xi, Ziming Xu, Shuang Liang, Hengyu Jin, Sibo Zhu, Wenyi Wu, Jinzhou Tang, Zijun Zhang, Songyao Jin, Xinyue Wang, Kun Zhou\*, Biwei Huang / Aether AI（aetherlabs.ai；创始人 Prof. Biwei Huang 黄碧薇，UCSD 助理教授）。† 推测为共同一作，\* 含义页面未注明。
- **原始链接：** <https://aetherlabs.ai/articles/real-world-autonomous-robotic-system-with-causality-driven-agent-and-world-model.html>
- **发表日期：** 2026-10-08（页面「Published 2026 · 10 · 08」；Blog 索引编号 11）
- **入库日期：** 2026-10-10
- **抓取方式：** `curl` 抓官方页静态 HTML（正文服务端渲染）；第 04 节「Experiments」由 `assets/blog/cris0/js/config.js` 动态注入，另行抓取该 JS 读取各实验标题 / 摘要 / 视频说明。视频本身未观看，仅取文字。Business Wire 原站 403，经 Morningstar 转载页（Playwright Chromium）取得全文。
- **一句话说明：** Aether AI 发布 **CRIS-0**（Causal Robotic Intelligence System）：由 **因果引导机器人智能体（Causality-guided Robot Agent）** 与 **因果世界模型（Causal World Model）** 组成，用 **显式因果状态变量** 表示任务进度，从少量遥操作示范自动分解阶段、生成验证条件与工具，再以「识别阶段 → 选工具 → 验证 → 重试 / 重规划」闭环执行；官方博文 **不含任何定量指标**，2 s 重规划、0.2 s 安全停、90%（18/20）个性化抓放等数字仅见于新闻稿与量子位报道（均为公司口径自报）。

## 其他来源

| 来源 | 链接 | 日期 | 说明 |
|------|------|------|------|
| 官方 News 条目 | <https://aetherlabs.ai/news.html> | 2026-10-08 | 「Aether AI Introduces CRIS-0, a Causality-driven Robotic Intelligence System for the Real World.」一段摘要，无数字 |
| 官方 Blog 索引 | <https://aetherlabs.ai/blog.html> | 2026-10-08 | 「Blog 11 · CRIS-0 · Causal AI · Robotics · ~10 min」 |
| Business Wire 新闻稿 | <https://www.businesswire.com/news/home/20261008233217/en/>（原站 403）；转载：<https://www.morningstar.com/news/business-wire/20261008233217/aether-ai-brings-causal-intelligence-into-the-physical-world> | 2026-10-08（Morningstar 标 6:30 PM） | 「Aether AI Brings Causal Intelligence Into the Physical World」；含 2 s / 0.2 s / 90% 数字与公司信息 |
| 量子位（QbitAI） | <https://www.qbitai.com/2026/10/502411.html> | 2026-10-09 | 「0.2秒急停、秒级重规划！因果智能走进真实世界」（作者 Jay）；含 9/10、18/20、0.2 s 等数字与架构解读 |

## 开源 / 项目页核查（步骤 2.5）

| 项 | 结论（截至 2026-10-10） |
|----|-------------------------|
| 项目页 / 技术报告 | 仅官方博文；Business Wire 称「A technical description of the system is available here」，推测即该博文。**无 arXiv / 论文 PDF** |
| 代码 | **未开源**：博文、News、新闻稿、量子位均无 GitHub 链接；博文 HTML 中无 github / huggingface / arxiv 字样（GitHub 组织页在本环境无法访问，未能直接检索） |
| 权重 | **未公开**：Hugging Face `author=aetherlabs` / `Aether-AI` / `AetherAI` / `aether-ai` / `AetherLab-AI` 模型列表均为空（2026-10-10 查询） |
| 数据 | **未公开**（遥操作示范数据、世界模型训练数据均未提及发布） |
| 可信度边界 | 公司博文 + 新闻稿 + 媒体报道；所有成功率、时延数字为 **公司自报**，且不在官方博文中出现；无试次协议、无基线对比、无第三方复核 |

## 核心摘录（归纳，非全文）

### 定位与主张（官方博文）

- 引言：真实部署要求机器人识别 **任务相关因果变量** 并推理它们随时间如何相互影响；能否应对复杂、变化环境主要取决于对任务进度与物理交互底层因果结构的建模。
- CRIS-0 = **Causality-guided Robot Agent**（围绕因果状态转移组织任务执行）+ **Causal World Model**（建模并预测动作的物理后果）。
- 给定新任务：分析 **少量示范** → 识别任务相关变量、把任务分解为阶段、定义验证准则、构建执行所需工具或策略；交互中持续从观测更新状态、验证中间结果，并按需调用 **规则工具、学习策略、导航模块或世界模型**。
- 三项能力：**Robustness**（识别任务相关扰动、与无关变化区分、回到受影响阶段恢复）；**Autonomy**（显式状态追踪进度、验证中间结果，自主决定继续 / 重试 / 重规划）；**Personalization**（任务结构、验证规则、工具可从示范与反馈构建与修正，适配不同用户偏好）。

### 2.1 智能体架构（Fig. 1）

- **输入：** 用户请求（例「Get me a drink that fits my fitness goals.」）；观测为 **头部、左、右三路相机**；工具反馈（verifier 结果、错误，如 `verify(grasp) → false`、`IKError: joint limit hit`）。
- **Causal Agent：** 维护 **Task state**（由观测与工具反馈持续更新），循环四步：Identify stage → Select tool → Verify outcome → Retry / replan。
- **Unified tool interface** 下的工具：
  - **Causal World Model**：预测动作的未来结果（+ Action → 策略）
  - **Policy**：接触丰富操作的策略模型
  - **Rule-based functions**：智能体生成的函数，使用 **SAM3、RGB-D、IK**
  - **SLAM navigation**：导航到任务所需位置
  - **Verifiers**：检查阶段完成并为重试 / 重规划提供反馈
- 图注：Causal World Model 预测动作结果，并 **作为接触丰富操作所用策略模型的基础**。
- 工具获取：从少量示范「extract tools and skills」；工具执行失败时 **自动修订工具并重试该阶段**。

### 统一因果状态与工具接口

- 每个任务阶段用一组 **关键因果变量** 表示，输入输出以统一格式组织。铺桌布角示例：`cloth_grasped`（是否真正抓住 → 决定能否开始拉，否则回退到抓取阶段）、`corner_offset_cm`（角点离目标距离 → 决定拉的方向 / 距离与阶段完成）、`slippage_detected`（拉动中是否滑脱 → 继续或重抓）。
- 工具输入 JSON：`stage` + `state` + `target` 三字段。
- 饮料示例：planner 选饮料 → `rule.position_gripper(target="coke")` 把夹爪移到附近 → `policy.skill(label="grasp")` 完成接触抓取；策略只负责局部技能，**无需理解整体任务或推断用户意图**。

### 2.2 结构化状态转移循环与任务图

- 连续循环：状态评估 → 工具执行 → 结果验证，逐步构建并精化 **任务图**（节点 = 可验证阶段，边 = 阶段间转移）；成功与失败尝试的反馈都会更新图。
- 任务分解四步：① **少量遥操作示范**，自动发现可复用执行模式；② planner 把任务拆成 **原子、可验证阶段**，区分自由空间运动与接触丰富交互；③ 每阶段定义 **可观测成功条件**，运行时由 **可执行验证脚本** 检查；④ 每阶段得到参考执行策略（生成函数或学习策略），以可调用工具暴露给 planner。
- 自由空间运动：从示范阶段末提取夹爪 / 手臂相对目标物体位姿 → 编码为可复用函数；运行时用 SAM3 + RGB-D 定位（如锅把手）→ 施加示范相对位姿 → IK 求关节构型。
- 接触丰富阶段（如稳定抓锅把手）：用 **自家策略模型** 作参考执行策略；按按钮等阶段两类工具都可，**同时暴露、运行时动态选择**。
- 清理桌布示例任务图：01 推布出桌沿（规则函数，判据：布伸出桌沿约 5 cm，循环直到满足）→ 02 定位夹爪（规则）→ 03 抓取并微调（技能策略；验证失败回 02 重抓）→ 04 折叠（技能策略）→ … → 07 放入篮子（规则）。
- 执行与验证：验证失败或出现意外阶段转移时，planner 重新评估状态，决定重试或修订计划；若失败来自工具本身（如 IK 解违反关节限位），**按错误反馈自动修订工具，并在成功执行后保留新实现**。
- 三点收益：(1) 跟踪阶段即可恢复（抓取时物体被移 → 任务回退到 approach → 重新调用定位函数）；(2) 每阶段先验证再前进，**误差不累积**；(3) 难任务被拆成每阶段配最合适工具的简单子问题，单一策略端到端难处理的操作得以完成。
- 执行轨迹示例（标注 Illustrative）：approach ✓ → grasp … → pot displaced, `verify(grasp_secure) → false` ✗ → state regressed → approach → IK 违反关节限位 ✗ → 依错误反馈 patch `rule.position_gripper` ✓ → grasp verified ✓ → retain revised tool ✓。
- 失败恢复路径：**以调整后的参数重试本阶段 / 回到更早阶段恢复必要条件 / 依反馈修订工具实现**；每次恢复也需验证；成功恢复路径并入任务图，修订后的工具留作复用。视频示例：丢弃卷尺包装、把梨放进碗。

### 03 因果世界模型

- 3.1 未来预测：通过任务相关因果变量预测未来物理演化，而非只做像素预测——**当前状态 → 因果变量转移 → 未来像素观测**；智能体可问「动作是否可能产生预期状态转移」，而不只是「视觉上会发生什么」。展示 6 段生成视频：递杯给人、叠衬衫、把蓝瓶放进抽屉、摆餐盘、把红球移到下层架、双手传递物体。
- 3.2 **Causal World Action Model 作为机器人策略**：在因果世界模型上增加 **动作模块**，骨干继续建模世界演化，动作模块把共享表示映射为可执行控制；策略可同时推理预期未来因果状态与动作侧变量后再输出控制。
- 博文 **未出现「CausalWM」字样**，也未说明与 2026-09-19 发布的 CausalWM 是否同一版本；新闻稿与量子位把它表述为 CausalWM。

### 04 真实世界评测（仅视频 + 文字说明，无数字）

| 类别 | 场景（官方视频说明） | 播放速度 |
|------|----------------------|----------|
| Robustness | 抓取前咖啡袋被移动；倒粉前磨豆机被移动 | 1× |
| Robustness | 盘中被加入一个苹果，机器人在微波前将其移除；**手靠近微波炉门，机器人后撤以免夹手** | 1× |
| Robustness | 强烈闪烁灯光下倒咖啡豆 | 1× |
| Autonomy | 一次连续运行整理整张杂乱茶几 | 10× |
| Personalization | 请求「拿饮料」时依用户偏好选不同饮料（2 个场景） | 1× |
| Context reasoning | 空罐扔掉、未开封罐放托盘；书放茶几、私人账单放抽屉 | 2× / 1× |
| Complex manipulation | 换桌布：折叠旧桌布放篮子；取新桌布铺平 | 1× |
| Generalization | 把蛋糕放盒子的技能迁移到把苹果放盘子 | 约 1× |

- 博文总结：在扰动、长程、欠指定请求、复杂操作、新物体五类设置上「reliably」完成任务——**无成功率、试次数或基线**。

### 新闻稿（Business Wire，2026-10-08）要点

- 称为 CRIS-0 的 **首次公开演示**，在家居场景驱动具身机器人；也是 Causality-guided Robot Agent 与 Causal World Model **首次作为集成系统** 运行。
- 显式因果表示 = 描述环境的变量 + 变量间及对机器人动作响应的因果关系；系统更新表示、预测潜在动作效果、选工具、**将结果与预测比较**，必要时重试或重规划。
- 闭环：**世界模型预测指导智能体选工具，执行结果回馈世界模型更新其状态**（此闭环表述官方博文未明确写出）。
- 前序研究：截至 2026-09，CausalWM 在 TriWorldBench 第一、PAI-Bench 机器人赛道第一；RSIAgent 无需额外训练在 OSWorld 2.0 与 Agents' Last Exam 上超过前沿闭源模型——均在仿真 / 软件环境中取得。
- 数字（自报，「according to the company」）：**咖啡豆倒粉 + 研磨任务** 中从人为干扰恢复，**平均约 2 s 重规划**；**微波炉场景** 检测到安全风险后 **平均 0.2 s 内停止或改计划**；**个性化抓放** 在需推理的复杂模糊提示试验中 **成功率 90%**。
- 公司信息：2026 年由 Prof. Biwei Huang 创立（UCSD 助理教授，因果发现领域 12 年）；累计融资约 **2000 万美元**；团队约 **20 人**；下一步扩展到预测（forecasting）与科学发现。

### 量子位（2026-10-09）要点

- 数字（报道口径，称「官方公布」）：**Coffee Preparation** 抗干扰测试——抓取时遭遇人为移位、光照突变，**平均 2 s 内** 识别因果变量变化并重规划，**10 次随机扰动 9 次有效恢复**；**Personal Pick and Place**——20 条需隐含推理的模糊指令（如「给我拿一瓶适合晨跑后喝的饮料」）**18 次** 精准抓放；关微波炉门时人手伸入 **0.2 s 急停 / 悬停**。
- 架构解读：Planner + **Unified Tool Interface**（规则运动函数、技能策略模型、SLAM 导航、验证器、因果世界模型 CausalWM）；任务图循环「**状态识别 → 工具选择 → 执行 → 验证 → 调整**」；**三级恢复**：先本阶段重试 → 不行重新规划 → 再不行提示人工介入；成功恢复路径沉淀进任务图。
- 安全：安全判断内嵌于每个因果阶段——关门阶段出现人手是破坏安全条件的危险变量，触发最高优先级阻断；递水阶段人手是交互目标变量，不会误停。
- 其他报道细节（官方博文未见）：机器人先抓取晃动饮料罐感知重量与液体状态再判断空罐；咖啡机被推开时追踪「把手相对位姿」；世界模型中先「排练」再调用 Action Head；长程整理「数十甚至上百步」。
- 对比叙事（报道观点）：VLA 等端到端模型在长程任务中误差累积、遇干扰不知发生了什么；常见 LLM / 多模态 Agent 方案把画面转文本再慢速推理，反应慢。
- 背景：RSIAgent 在 OSWorld 2.0 上 Partial Score **78.98%**（不更新参数）；CausalWM 登顶 TriWorldBench；称这两层是本次 Demo 的主要贡献者；另两层（模块化神经架构、Causation Transformer）仍在推进。黄碧薇师承 Clark Glymour、Peter Spirtes、Bernhard Schölkopf、Kun Zhang（报道表述）。

## 来源间差异与核对

| 项 | 官方博文 | Business Wire | 量子位 |
|----|----------|---------------|--------|
| 咖啡任务恢复 | 视频：咖啡袋 / 磨豆机被移、闪光灯；**无数字** | 咖啡豆倒粉 + 研磨，平均 **2 s** 重规划 | 平均 **2 s**，**9/10** 有效恢复；被移物描述为「咖啡机」「把手相对位姿」 |
| 安全停 | 视频：手靠近微波炉门 → 后撤；**无数字** | 微波炉场景平均 **0.2 s** 内停止或改计划 | 关门时 **0.2 s** 急停 |
| 个性化抓放 | 视频 2 个场景；**无数字** | **90%** 成功率 | **18/20**（=90%，与新闻稿一致） |
| 恢复层级 | 重试（调参）/ 回到早期阶段 / 修订工具 | 调整、重试、重规划 | 重试 → 重规划 → 人工介入（三级） |
| 世界模型名称 | 「Causal World Model」（未称 CausalWM） | CausalWM 为前序研究 | 直接称 CausalWM |

- 本次抓取的三份来源 **均未出现 0.1 s**；安全停数字两份媒体一致为 0.2 s。
- 9/10 只见于量子位；新闻稿只给 2 s 均值。

## 对 wiki 的映射

- [aether-cris-0](../../wiki/entities/aether-cris-0.md) — 本篇升格实体页
- [aether-ai](../../wiki/entities/aether-ai.md) — 公司入口页
- [paper-causalwm](../../wiki/entities/paper-causalwm.md) — CRIS-0 内的因果世界模型
- [aether-rsiagent](../../wiki/entities/aether-rsiagent.md) — 媒体称其为智能体层的前序研究

## 可信度与使用边界

- 官方博文定性描述架构与视频；所有定量数字来自新闻稿与媒体，且标明「according to the company」——视为 **自报**，评测协议、试次分布、失败案例均未公开。
- 机器人硬件平台型号、planner 所用基础模型、策略模型训练数据规模均 **未披露**。
- 无对照基线（无与 VLA 或 LLM-agent 方案的同任务数字比较）。

## Citation

```bibtex
@misc{aether2026cris0,
  author = {Mao, Lingjun and He, Lukun and Cao, Jinglin and Xu, Wenpeng and others and Zhou, Kun and Huang, Biwei},
  title = {Building Real-world Autonomous Robotic System with Causality-driven Agent and World Model},
  howpublished = {Aether AI Blog},
  year = {2026},
  note = {https://aetherlabs.ai/articles/real-world-autonomous-robotic-system-with-causality-driven-agent-and-world-model.html}
}
```
