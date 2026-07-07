# autoresearch（karpathy/autoresearch）

> 来源归档

- **标题：** autoresearch
- **类型：** repo
- **来源：** Andrej Karpathy
- **链接：** https://github.com/karpathy/autoresearch
- **父项目：** [nanochat](https://github.com/karpathy/nanochat)（本仓为单 GPU 简化训练栈）
- **入库日期：** 2026-06-20
- **一句话说明：** 让编码代理在固定 5 分钟墙钟预算内自主修改 `train.py`、训练 nanochat 式 GPT、以 **val_bpb** 决定去留并循环实验；人类主要迭代 `program.md`（轻量「研究组织技能」）而非直接改 Python。
- **为什么值得保留：** 把 [AI Auto-Research](../../wiki/concepts/ai-auto-research.md) 综述中 **S3 编码与实验** 落到可复现的最小闭环：单文件可编辑面、固定可比预算、单一验证指标、人机分工（`program.md` vs `train.py`）。与 [LLM Wiki（Karpathy 模式）](../../wiki/references/llm-wiki-karpathy.md)、[Superpowers](../../wiki/entities/superpowers-obra.md)（`SKILL.md` 技能契约）形成 **研究自动化 / 知识编译 / 工程技能** 三角参照。
- **沉淀到 wiki：** 是 → [`wiki/entities/karpathy-autoresearch.md`](../../wiki/entities/karpathy-autoresearch.md)

## README 要点（归纳）

- **叙事定位：** 科幻式开场 + 务实目标 — 给代理一个 **真实但小** 的 LLM 训练环境，让其 **通宵自主实验**；早晨查看实验日志与（希望）更优模型。
- **核心分工：**
  - **`prepare.py`** — 常量、一次性数据与 BPE tokenizer 准备、运行时工具（dataloader、eval）；**不修改**。
  - **`train.py`** — 完整 GPT、优化器（Muon + AdamW）、训练环；**代理唯一可改文件**（架构、超参、batch、优化器等皆可动）。
  - **`program.md`** — 单代理基线指令；**人类迭代** 的「研究组织代码」，README 称其为超轻量 **skill**。
- **实验契约：**
  - 固定 **5 分钟墙钟** 训练预算（不含启动/编译），约 **12 次/小时**、睡眠约 **100 次** 实验量级。
  - 主指标 **val_bpb**（validation bits per byte）— 越低越好，与词表规模无关，便于跨架构公平比较。
  - 代价：跨平台算力不可直接横向对比结果。
- **运行要求：** 单张 NVIDIA GPU（README 在 H100 测过）；Python 3.10+；[uv](https://docs.astral.sh/uv/) 管理依赖；`uv run prepare.py` 一次性准备后 `uv run train.py` 手测，再进入代理自主模式。
- **代理启动示例：** 在仓库内启用 Claude/Codex 等（README 建议关闭权限拦截），提示阅读 `program.md` 并 kick off experiment setup。
- **设计选择：** 单文件 diff 可审；自包含（无分布式、无重配置）；小平台 fork 指南（TinyStories、降 `vocab_size`/`DEPTH`/`MAX_SEQ_LEN` 等）。
- **协议：** MIT。

## 对 wiki 的映射

- 升格实体页：[karpathy/autoresearch](../../wiki/entities/karpathy-autoresearch.md)
- 交叉补强：
  - [AI Auto-Research（概念）](../../wiki/concepts/ai-auto-research.md) — S3 最小可运行实例；Explore→Execute→Verify 分层。
  - [Andrej Karpathy](../../wiki/entities/andrej-karpathy.md) — nanochat 教育栈延伸与 LLM 时代实验自动化。
  - [LLM Wiki（Karpathy 模式）](../../wiki/references/llm-wiki-karpathy.md) — `program.md` 作为可版本化「组织上下文」与 ingest 文件契约同构。
  - [Superpowers（obra）](../../wiki/entities/superpowers-obra.md) — `program.md` ≈ 单文件 skill；本仓偏 **ML 实验搜索** 而非软件交付 TDD。
  - [Darwin Skill](../../wiki/entities/darwin-skill.md) — 将 keep/revert 棘轮映射到 `SKILL.md` 优化；与 [Nuwa](../../wiki/entities/nuwa-skill.md) / [Cangjie](../../wiki/entities/cangjie-skill.md) 造 skill 闭环。

## 参考来源（原始）

- GitHub README：<https://github.com/karpathy/autoresearch>（2026-06-20 抓取要点）
- Karpathy 推文：<https://x.com/karpathy/status/2029701092347630069> · <https://x.com/karpathy/status/2031135152349524125>
