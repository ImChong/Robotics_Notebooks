# HarnessEval: The Era of Harness for Benchmarking（MirroS Blog）

> 来源归档（blog / MirroS 官方）

- **标题：** HarnessEval: The Era of Harness for Benchmarking
- **类型：** blog
- **作者：** MirroS Team
- **原始链接：** <https://mirros.ai/blog/harnesseval>
- **发表日期：** 2026-08-17
- **入库日期：** 2026-08-18
- **抓取方式：** 官方博客页面直接抓取（WebFetch）；与项目页、arXiv:2608.16859、GitHub 交叉核对
- **一句话说明：** 把 LLM 生态的 **Harness** 概念接到世界模型评测：评测是可执行的 agentic 工作流（取证、工具、推理），而不是静态 rubric；并以 HarnessEval-W 作为第一份交互式世界模型实例。

## 项目页与开源核查（步骤 2.5）

核查日 **2026-08-18**：

| 入口 | 结果 |
|------|------|
| 本博客 | 概念叙事 + 三轴 + 九技能示意图 + 案例构建流水线；文末开源邀请 |
| 项目页 <https://mirros-lab.github.io/HarnessEval-W> | 链到 Code / 论文；技能数写 **11** |
| GitHub `MirroS-Lab/HarnessEval-W` | **已开源、可运行** 评测 CLI 与 11 个 skill 模块 |
| Hugging Face | 博客未给数据集卡；README TODO 仍待勾 |

**开源结论：评测系统已开源；全量案例 HF 托管待发布。** 博客技能图列 9 个高层名，以仓库 `SKILLS`（11）与项目页为准。

## 核心摘录（归纳，非全文）

### 为什么现有世界模型榜「不够说服人」

- Physical RSI 场景下，物理因果、几何一致、观测真实感仍脆弱。
- 人很容易看出生成伪影，现有基准从未成功自动化这一能力。
- 分数既不能解释也不能核验，也给不出失败定位。

### Harness 是什么（相对「套一层评测脚本」）

- 不是 code wrapper，而是把复杂人类工作流（取证、用工具、推理）形式化为可跑的 agentic 脚手架。
- 人类评生成世界：定位物体、跟踪 object permanence、核因果与几何约束——这套工作流可以被 harness。
- HarnessEval 像侦探：拆问题、配工具、对每个案例给出可检查的推理序列。

### 三轴（与论文一致）

- **Observation Quality：** 感知连贯、结构合理、视频真实感。
- **Transition Correctness：** Exploratory / Intentional / Physical。
- **World Persistence：** Drift Resistance / Revisit Consistency / Offscreen Evolution。持久性 **不等于** 处处冻结：不变量保持，动态量随动作与时间一致演化。

### 分层工作流

Skill 选择（只看案例上下文）→ Skill 分解（可测子问题 + 子代理）→ 证据树。示例：碰撞可拆成跟踪 bbox、核时序相交、估速度。Intentional Change Verifier 再拆八个子问题。

博客图示九技能：Render Quality / Physical Plausibility / Viewpoint Trajectory / Intentional Change / Physical Response / Physical Dynamics / Drift Degradation / Return Consistency / Offscreen Evolution。仓内另有 motion quality 与 appearance consistency。

### 案例构建

Scene taxonomy 采样 → 指定评测轴 → Imagen Agent 合成首帧 → 图像 grounding 的 Planner 出动作 → Validator 核环境/前景/中景/动作是否自洽；失败回采样。

### 下一步（博客 § What's Next）

1. **Test-time scaling：** 评测器随被评模型变强，用更多计算做更深搜索与多步核验。
2. **Scaling skill libraries：** 技能库随场景与物理保真度增长。
3. **Recursive self-improved agentic benchmark：** OOD 时评测评测器自身的技能缺口，外扩或自探索写回技能库。

## 对 wiki 的映射

- [`wiki/entities/paper-harnesseval-w.md`](../../wiki/entities/paper-harnesseval-w.md)
- [`sources/papers/harnesseval_w_arxiv_2608_16859.md`](../papers/harnesseval_w_arxiv_2608_16859.md)
- 评测选型 ② 层：[embodied-eval-benchmark-selection-loop](../../wiki/queries/embodied-eval-benchmark-selection-loop.md)
