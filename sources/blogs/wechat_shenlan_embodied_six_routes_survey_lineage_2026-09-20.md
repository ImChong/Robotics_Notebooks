# 六大具身路线详解：模块化、技能编排、IL、RL、VLA、世界模型，到底在"吵"什么。。。

> 来源归档（blog / 微信公众号）

- **标题：** 六大具身路线详解：模块化、技能编排、IL、RL、VLA、世界模型，到底在"吵"什么。。。
- **类型：** blog
- **作者：** 深蓝具身智能（编辑｜阿豹；审编｜具身君）
- **原始链接：** https://mp.weixin.qq.com/s/iyzL2yLzIqsIRergN3qS_Q
- **发表日期：** 2026-09-20（《具身智能基础》专栏第 13 篇；create_time 未单独核）
- **入库日期：** 2026-09-20
- **抓取方式：** WebFetch 直拉 `mp.weixin.qq.com` 正文（本环境无预装 wechat-article-for-ai；Jina Reader 对公众号常返回验证页）
- **原始抓取落盘：** [`sources/raw/wechat_shenlan_embodied_six_routes_survey_lineage_2026-09-20.md`](../raw/wechat_shenlan_embodied_six_routes_survey_lineage_2026-09-20.md)
- **一句话说明：** 沿六条产业路线（模块化 / 技能编排 / IL / RL / VLA / WM）串读 12 篇经典综述，强调「老问题未消失、技术单位变大」；与 2026-09-04「六条窟窿」文互补——本篇偏历史脉络与补课书单，不重复产业卡点表。

## 核心摘录（归纳，非全文）

### 开场：数据是症状，不是答案

- 文引行业估算：通用机器人 **70%–80%** 基础成功率或需 **1 亿小时** 示范/交互；「开箱即用」级或到 **千亿小时**（公众号转述，非本库核实）。
- 过去 18 个月 **100+** 开源具身基础模型；截至 2026-08 **30+** 国内公司宣称做世界模型（转述）。
- 核心追问：**机器人到底该从数据里学什么？** 六条路线是对「从什么里学」的六次不同回答，核心诉求（任务分解、技能组合、示范、试错、符号接地、预测未来）并未因新缩写消失。

### 六条 × 代表综述（文内 12 篇）

| 路线 | 文内经典综述锚点 | 文内 continuity 论点 | 本库节点 |
|------|------------------|----------------------|----------|
| **模块化规划** | 2021 TAMP 形式化；2024 FM 在机器人栈中的位置综述 | Task vs Motion 两尺度；层间翻译丢信息；FM 改「模块里装什么」而非消灭模块 | [轨迹优化 / TAMP](../../wiki/methods/trajectory-optimization.md)、[ScheduleStream](../../wiki/entities/schedulestream.md) |
| **技能编排** | 2022 Behavior Tree 综述；2024 LLM for Robotics | Skill 早存在；变化的是调度员（BT → LLM）；可混合 PDDL + 经典规划器 | [行为树 × VLA 编排](../../wiki/concepts/behavior-tree-vla-orchestration.md)、[LLM 控制接口](../../wiki/concepts/llm-robotics-control-interfaces.md) |
| **模仿学习** | 2009 LfD 综述；2026 FM for Manipulation 综述 | 示范从「教一项技能」→ 基础模型训练资源；分布与 embodiment 对齐问题放大 | [Action chunking](../../wiki/methods/action-chunking.md)、[Diffusion Policy](../../wiki/methods/diffusion-policy.md) |
| **强化学习** | 2019 continuous control RL tour；2025 real-world DRL successes | 真机 interaction 成本与安全；从「仿真学会」到「真机验证成熟度分级」；WM 作 learned simulator | [RL](../../wiki/methods/reinforcement-learning.md)、[Sim2Real](../../wiki/concepts/sim2real.md) |
| **VLA** | 2016 Symbol Emergence；2026 VLA survey (TNNLS) | Grounding 未消失；从「身体建词义」到「互联网 VLM 再接地到 action space」 | [VLA](../../wiki/methods/vla.md)、[VLA 综述归档](../papers/hmi_p071_vla-survey-embodied.md) |
| **世界模型** | 2020 MBRL survey；2026 WM for Robot Learning 综述 | 预测未来再行动并不新；video WM 把状态转移可视化；角色含 policy / simulator / data gen | [生成式 WM](../../wiki/methods/generative-world-models.md)、[WM 综述](../papers/wm_robot_survey_arxiv_2605_00080.md) |

### 收束（文内判断）

- **技术单位变大**：VLM/LLM/VLA/WM 通过共享表示与更大数据连接原分离模块；物理摩擦、碰撞、噪声、 embodiment 差异不随参数量消失。
- **补课顺序建议**：先经典综述建立 Planning / Skill / IL / RL / Grounding / MBRL 问题意识，再读 FM / VLA / 新一代 WM。
- **产业共识（文内，工程判断）**：短期 VLA 仍是控制主体，WM 辅助规划/数据/仿真；长期双向融合。

## 对 wiki 的映射

- **主写回：** [六条路线 × 12 篇综述补课线](../../wiki/queries/embodied-six-routes-survey-lineage.md)（query 产物）
- **交叉（不重复）：** [六条路线的窟窿](../../wiki/queries/embodied-six-routes-holes.md)（同系列 2026-09-04 文，偏 2026 卡点与分层）
- **正交坐标：** [五大范式](../../wiki/comparisons/robot-learning-five-paradigms-taxonomy.md)、[五大模型族选型闭环](../../wiki/queries/embodied-fm-taxonomy-loop.md)
- **本次未**为 12 篇综述逐条新建 `sources/papers/`：站内已有 VLA / WM 综述归档；其余标「文内引用，待单篇 ingest」

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 六条路线 × 12 综述对照表与 wiki query 写回
- [ ] 12 篇综述逐条 arXiv / DOI 核对与 paper 归档（后续 lint 跟进）
