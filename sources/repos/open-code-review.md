# Open Code Review（alibaba/open-code-review）

> 来源归档

- **标题：** Open Code Review（OCR）
- **类型：** repo
- **来源：** 阿里巴巴（Alibaba Group）
- **链接：** https://github.com/alibaba/open-code-review
- **官方站：** https://open-codereview.ai
- **npm：** `@alibaba-group/open-code-review`（全局命令 `ocr`）
- **入库日期：** 2026-09-19
- **协议：** Apache-2.0
- **一句话说明：** 阿里巴巴内部 AI 代码评审助手开源版：以 **确定性工程管线**（精确选文件、智能分包、模板规则匹配、评论定位/反思）约束评审过程，再叠加 **带 tool-use 的 Agent** 做动态上下文检索；支持 `ocr review` / `ocr scan` / Delegation Mode 与多 harness 插件。
- **为什么值得保留：** 与本知识库维护者常用的 **Cursor / Codex / Claude Code** 及 [Superpowers](../../wiki/entities/superpowers-obra.md) 工作流中的 **requesting-code-review** 环节直接相邻；对「如何把 code review 从纯 prompt skill 升级为可审计管线」有对照价值。
- **沉淀到 wiki：** 是 → [`wiki/entities/open-code-review.md`](../wiki/entities/open-code-review.md)

## README 要点（归纳）

- **规模背书：** 内部两年服务数万开发者、识别数百万缺陷后开源；Trendshift / OpenSSF Gold 等徽章（以 README 为准）。
- **核心痛点（相对通用 Agent + Skills）：** 大 diff **覆盖不全**、评论 **行号漂移**、纯自然语言 skill **质量不稳定**。
- **混合架构：**
  - **确定性工程：** 文件选择、相关文件 bundling（子 agent 分治）、模板引擎规则匹配、外部定位与反思模块。
  - **Agent：** 场景化 prompt + 从生产 trace 蒸馏的 review 专用 toolset（读全文件、代码搜索、跨文件上下文）。
- **主要命令：** `ocr config provider|model`；`ocr review`（workspace / `--from`–`--to` / `--commit`）；`ocr scan`（全文件审计）；`ocr delegate`（Delegation Mode）；`ocr session list` / `--resume`；`--format json --output` 供宿主 agent 消费。
- **基准数据：** [AACR-Bench on Hugging Face](https://huggingface.co/datasets/Alibaba-Aone/aacr-bench)。
- **实现语言：** Go（`cmd/opencodereview` + `internal/*`）；npm 包装为跨平台 CLI 分发。
- **前置依赖：** Git ≥ 2.41。

## 与本站 sources 的其它锚点

- 项目页核查：[`sources/sites/open-codereview-ai.md`](../sites/open-codereview-ai.md)
