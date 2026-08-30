# Meta-Ctrl（语法/语义解耦约束解码）

> 来源归档（ingest）

- **标题：** Meta-Ctrl: Guaranteed Plan Generation by Decoupling Syntactic and Semantic Constraints
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.22149>
- **机构：** 加州大学洛杉矶分校（UCLA）；密歇根州立大学（Michigan State University）
- **作者：** Gwen Yidou-Weng、Edward Sun、Tianyi Ma、Metin Alp Dogan、Benjie Wang、Allen Peng、Guy Van den Broeck、Yuchen Cui
- **项目页：** <https://meta-ctrlg.github.io/>
- **入库日期：** 2026-08-30
- **一句话说明：** 用元令牌把机器人计划的语法约束（token 级）与语义约束（动作级）精确因式分解，保证合法同时保留 LM 常识。

## 核心摘录（MVP）

### 1) 软约束无保证，符号规划丢常识

- **摘录要点：** LLM 计划常违反可执行语法/语义。软方法（affordance、grounded decoding）无保证；LLM+P 类符号规划丢掉语言模型常识。
- **对 wiki 的映射：**
  - [Meta-Ctrl](../../wiki/entities/paper-meta-ctrl.md)

### 2) 两级因式分解

- **摘录要点：** 语法 \(\gamma\) 用 DFA 在 token 级保证动作名与参数格式；语义 \(\beta\) 在约 132 个 meta-token 上跟踪前置条件、目标与顺序。精确分解把受约束解码内存从 **>107 TB 降至 <2 GB**（约 67,000×），计算约 1,900×。
- **对 wiki 的映射：**
  - [Meta-Ctrl](../../wiki/entities/paper-meta-ctrl.md)

### 3) 评测数字

- **摘录要点：** Llama-3-8B VirtualHome Action Sequencing 任务成功率 **21.3→88.7**，超过 GPT-4o / Claude-3.5 / o1-preview。WAH-NL（LoTa-Bench）Llama 3.1 8B：SSR **0.705**、Exec **1.000**，超过 GPT-4 的 0.342。xArm7 桌面：计划按构造 100% 满足前置与目标；失败发生在感知/抓取。
- **对 wiki 的映射：**
  - [Meta-Ctrl](../../wiki/entities/paper-meta-ctrl.md)

### 4) 开源状态（截至 2026-08-30）

- **摘录要点：** **未开源**。项目页有方法与表；`meta-ctrlg/meta-ctrlg.github.io` 仅为 Pages。

## 当前提炼状态

- [x] 项目页与 arXiv 摘要对齐
- [x] wiki 映射：`wiki/entities/paper-meta-ctrl.md` 新建
