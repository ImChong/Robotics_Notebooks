# HarnessEval-W: Agentifying the Evaluation of Visual Worlds（arXiv:2608.16859）

> 来源归档（ingest）

- **标题：** HarnessEval-W: Agentifying the Evaluation of Visual Worlds
- **缩写 / 框架：** **HarnessEval-W**（评测套件与 CLI：`harnesseval`）；概念层简称 **HarnessEval**
- **类型：** paper / benchmark / agentic-evaluation / interactive-world-model
- **arXiv：** <https://arxiv.org/abs/2608.16859>（Submitted 2026-08-17；PDF：<https://arxiv.org/pdf/2608.16859>）
- **项目页：** <https://mirros-lab.github.io/HarnessEval-W> — 归档见 [`sources/sites/harnesseval-w-github-io.md`](../sites/harnesseval-w-github-io.md)
- **代码：** <https://github.com/mirros-lab/harnesseval-w>（README 宣称 Apache-2.0）— 归档见 [`sources/repos/harnesseval-w.md`](../repos/harnesseval-w.md)
- **全文 Blog：** <https://mirros.ai/blog/harnesseval> — 归档见 [`sources/blogs/mirros_harnesseval.md`](../blogs/mirros_harnesseval.md)
- **作者 / 团队：** MirroS Team（BibTeX：`{MirroS Team}`）
- **机构：** 镜界（MirroS）/ MirroS-Lab
- **入库日期：** 2026-08-18
- **一句话说明：** 把 LLM 生态的 **harness** 接到交互式世界模型评测：按案例路由技能、分解子问题、子代理取证、父代理校验，产出可审计证据树；330 例 × 18 模型，Intentional 维与人类 Bradley–Terry 排序 Spearman ρ=0.93。

## 开源状态（步骤 2.5）

核查日：**2026-08-18**（项目页 / GitHub README / 仓目录 / 博客 TODO）。

| 产物 | 状态 |
|------|------|
| 评测代码 + CLI（`harnesseval eval / plan / generate / verify`） | **已开源**；`src/harnesseval/` 含 planner、runner、11 个 skill、metric backends |
| 固定 plans + 捆绑 demo（`runs/example/results_example`） | **已开源**；README 称装好三环境后即可评 demo |
| `benchmark/plans`、`benchmark/initial_observations` | **仓内已发布** |
| 330 例全量 / 子集到 Hugging Face | **待发布**（README TODO 未勾） |
| 托管提交评测服务 | **待发布** |
| GitHub License 字段 | README 写 Apache-2.0；API `license=null`（根目录未见 GitHub 识别的 LICENSE 元数据） |

**结论：** **已开源、可运行评测管线**（代码 + 固定计划 + demo）；**全量案例与 HF 托管尚未发布**。勿写成「数据集已上 HF 活榜可提交」。项目页 2026-08-18 V1 Leaderboard 标注 Coming Soon。

## 摘录 1：问题与 harness 范式（§1）

- **目标：** 基准不能只给标量分；世界模型评测要能解释物理、因果与世界状态是否正确演化。
- **现状：** 现有基准多为固定量纲暴力打分，没有可检查、可验证的推理链，也难定位失败原因。
- **harness：** 来自 LLM 生态——不只是代码包装，而是把「取证 / 用工具 / 推理」这类人类工作流形式化为可执行 agent 脚手架。人类评生成世界时会定位物体、跟踪恒常性、核因果与几何；该工作流可被 harness。
- **HarnessEval-W：** 按案例上下文路由技能 → 分解可测子问题 → 子代理带诊断工具取证 → 父代理校验并汇总。产出 **evidence tree**（测了什么、哪件工具提供视觉 grounding、完整逻辑链）。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-harnesseval-w.md`](../../wiki/entities/paper-harnesseval-w.md)；挂 [评测基准选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md) ② 层（交互式开放域，非操纵 EEF 轴）。

## 摘录 2：交互式世界模型分解与三轴八设定（§3.2–3.3）

形式化（式 1）：未来观测分布由隐状态边缘化得到——观测似然 \(S(o_i\mid s_i)\)、动作条件转移 \(T(s_i\mid s_{i-1},a_{i-1})\)、由历史推断的初态 \(P(s_0\mid o_{-T:0})\)。动作可以是探索性视角变化、对指定实体/事件的意图性改变，或物理干预。

| Evaluation Axis | Detail settings | 核心世界状态问题 |
|-----------------|-----------------|------------------|
| Observation Quality | Render Quality / Physical Observation | 视频是否可读、稳定；帧是否结构/物理上说得通 |
| Transition Correctness | Exploratory / Intentional / Physical | 视角变化、指定目标变化、物理干预是否产生对应动力学响应 |
| World Persistence | Drift / Revisit / Offscreen | 长程不变量是否存活；离开再回是否兼容；看不见的内生过程是否继续 |

**对 wiki 的映射：** 与 [WorldScore](../../wiki/entities/paper-worldscore.md)（相机可控 next-scene）和 [EWMBench](../../wiki/entities/ewmbench.md)（操纵场景守恒/末端/语义）分轴，不要混读。

## 摘录 3：分层 agentic 评测（§3.4）

- **Skill routing 只依赖案例上下文与评测意图，不依赖被评模型** → 同案例对所有模型问同一组问题。
- 高层 skill 再拆成可测子问题；例：Intentional Change Verifier 拆成目标可见、过渡可见、意图变化、目标特异性、终态、锚点保持、无额外事件、可判定等子代理。
- 父 skill 校验证据树后再出案例分；低分可追溯到具体失败（错目标 / 掺入无关事件等）。
- 代码仓 `src/harnesseval/protocols.py` 登记 **11** 个 `SKILLS`（博客图示 9 个高层名；项目页写 11）。观测侧四件：`render_quality_inspector`、`motion_quality_inspector`、`appearance_consistency_inspector`、`physical_plausibility_inspector`。

**对 wiki 的映射：** 实体页画「案例 → planner → skills → 子代理 → validator → 证据树」流程图与 CLI 运行时序图。

## 摘录 4：案例构建与规模（§4）

- **场景分类六轴：** Environment / Foreground / Midground / Scene Density / Appearance / Perspective。
- **Probe family 六类** 对应 Transition + Persistence（Observation Quality 每例都评，不单独成 family）。
- **Agentic authoring：** Image Generator → Image-grounded Planner（不得改 family、不得引入图中不存在的实体）→ Case Validator；失败回采样。
- **发布规模：** 330 例。项目页快照：108 exploratory、51 intentional、66 physical；34 drift、34 revisit、37 offscreen。

## 摘录 5：实验要点（§5）

- **18 模型**，按接口分组：Prompt I2V / Native action / Camera pose；交互翻译成各模型原生输入，案例意图保持不变。
- **Overall** = 330 例案例级算术平均。Obs 两维对全部 330 例平均；其余维只在对应 probe family 上平均。
- **主榜前列（Overall）：** Seedance 2.0\* 75.5；Wan 2.7\* 75.0；Kling 3.0\* 74.4；MiniMax H3 74.3。闭源标 \*。
- **分项冠军不重合：** Wan 2.7 领 Intentional/Physical；Seedance 领 Drift；HY-WorldPlay 1.5 领 Revisit；SANA-WM 领 Offscreen。
- **人对齐（9 模型、5000 A/B → Bradley–Terry）：** Intentional ρ=0.93（τ=0.82）；Physical ρ=0.87（τ=0.74）。
- **对照 WBench（同视频、同 GPT-5.5 后端）：** Physical 成对准确率 31.9%→71.7%，平局 52.2%→1.8%；Intentional 60.2%→77.8%，平局 36.1%→11.1%。三次重复包络比 WBench 窄 **4.9×**。
- **轴相关：** Render Quality 与 Physical Observation r=-0.04；Intentional↔Physical r=0.98；Exploratory 与二者近无关（r≈-0.15/-0.18）。
- **微调位移：** Wan 2.2→DreamX-World、HunyuanVideo 1.5→HY-WorldPlay 1.5 均 **Revisit↑、Intentional/Physical↓**（论文归因微调数据偏探索轨迹）。

**对 wiki 的映射：** 读「可解释性 + 人对齐 + 接口族能力重分配」，不要只抄 Overall 冠军。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-harnesseval-w.md`**（流程总览 + 源码运行时序图 + 结论）。
- 新建 `sources/sites/`、`sources/repos/`、`sources/blogs/`。
- 交叉更新评测枢纽 / Query ② 层、WorldScore、EWMBench、生成式世界模型；ABot-World-0 榜上有分可回链。
