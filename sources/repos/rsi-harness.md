# RSI-Harness（RSIH）

> 来源归档

- **标题：** RSIH — versionable, shareable agent harness
- **类型：** repo
- **组织：** CosmosMind-ai
- **链接：** <https://github.com/CosmosMind-ai/RSI-Harness>
- **Hugging Face：** <https://huggingface.co/CosmosMind/RSI-Harness>
- **论文：** [MetaRSI-v1](https://www.cosmosmind.ai/research/metarsi-v1) / [arXiv:2609.06396](https://arxiv.org/abs/2609.06396)
- **项目页：** <https://cosmosmind.ai/> · <https://www.cosmosmind.ai/research/metarsi-v1>
- **入库日期：** 2026-09-14
- **一句话说明：** 基于 [Pi coding agent](https://www.npmjs.com/package/@earendil-works/pi-coding-agent) 的 **Genome** 配置层：把 system prompt、tools、skills、MCP、runtime 等 12 组件收成可版本化目录；内置 `harness-rsi`（GEE 从会话史生成 Genome）与 `paperlab` 示例。
- **沉淀到 wiki：** [`wiki/entities/rsi-harness.md`](../../wiki/entities/rsi-harness.md)、[`wiki/entities/paper-metarsi-v1.md`](../../wiki/entities/paper-metarsi-v1.md)

---

## 开源状态（步骤 2.5）

| 项 | 核查结论（2026-09-14） |
|----|------------------------|
| **GitHub** | [CosmosMind-ai/RSI-Harness](https://github.com/CosmosMind-ai/RSI-Harness) 公开；`install.sh` + `npm run check` 可构建；约 51 stars |
| **Hugging Face** | [CosmosMind/RSI-Harness](https://huggingface.co/CosmosMind/RSI-Harness) 镜像 README；声明 **无** benchmark / data-generation / training / evaluation 代码 |
| **许可** | 根目录 **无 LICENSE 文件**；`package.json` 未声明 license 字段 — 使用前以仓库最新声明为准 |
| **结论** | **已开源（Harness-RSI 运行时 + Genome 工具链）**；**非** MetaRSI 全栈（Model-RSI / Data-RSI 训练环未随仓发布） |

---

## 核心定位

RSIH = **Pi 不变 + Genome 适配层**。无 `--genome` 时行为与 `pi` 一致（测试守卫），仅配置目录改为 `~/.rsih`。有 Genome 时按 12 组件 patch Pi 的公开配置面（inherit-by-default；越界写入 load 时失败）。

**自指 RSI 落点：** `harness-rsi` Genome 的 charter 在 `instructions`、方法论在 skill、交互工具在 extension — **不 fork Pi Core**；`src/` 无专供 `harness-rsi` 的分支逻辑。

---

## 仓库入口

| 组件 | 说明 |
|------|------|
| 安装 | `git clone … && ./install.sh`；Node **≥ 22.19**；`~/.local/bin/rsih` |
| 无 Genome | `rsih` / `rsih --resume` / `rsih -p "…"` — 等同 `pi` |
| 启动 Genome | `rsih :paperlab` / `rsih --genome harness-rsi` / `rsih +name` |
| GEE | `gee` ≡ `rsih :harness-rsi` — 从 RSIH/Pi/Claude Code session 聚合证据生成 Genome |
| 管理 | `rsih genome list|show|validate|install` |
| 开发 | `npm run check`（typecheck + test + build）；`npm run build:binary` |
| 文档 | `docs/README.md`；组件契约 `docs/genome/components/` |

### 内置 Genome

| ID | 用途 |
|----|------|
| `paperlab` | 论文实验 harness 示例（scout / bootstrap / run-ops skills） |
| `harness-rsi` | 从会话史 **生成** 其它 Genome（Harness-RSI 演示） |

### 12 Genome 组件（互斥字段所有权）

`instructions` · `tools` · `skills` · `commands` · `model` · `runtime` · `policies` · `integrations` · `appearance` · `settings` · `keybindings` · `resources`

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-metarsi-v1](../../wiki/entities/paper-metarsi-v1.md) | MetaRSI 论文；Harness-RSI 算子 |
| [sol-pi](../../wiki/entities/sol-pi.md) | 同在 Pi harness 上的 auto-research / 效率扩展 |
| [karpathy-autoresearch](../../wiki/entities/karpathy-autoresearch.md) | 固定 eval 环对照 |
| [deepseek-harness](../../wiki/entities/deepseek-harness.md) | 另一类插件化 agent 运行时 |
| [recursive-self-improvement](../../wiki/concepts/recursive-self-improvement.md) | RSI 概念框架 |
