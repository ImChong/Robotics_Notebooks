# PaperSpine5（官方上游仓库）

- **类型：** repo / 论文写作系统
- **原始仓库：** https://github.com/WUBING2023/PaperSpine
- **上游身份：** GitHub REST `fork: false`；非 michae1liu/paperspine Fork
- **核验版本：** `42312b8d9c76`（main，2026-10-14）
- **入库日期：** 2026-11-11
- **许可：** MIT
- **发布状态：** 官方 README 对应 PaperSpine5，链接 v0.4.0-alpha.3；仍为 Alpha
- **开源状态：** 已发布 SKILL、Python 脚本、Web 接入层及安装包，部分功能需本地依赖（如 Pandoc/LaTeX）与配置
- **核对范围：** 已通读官方 `README.md`、`src/skill/SKILL.md` 主技能原文及 `src/skill/references/paper-spine-production-protocol.md` 全部 17 步质量协议；已通读 `src/skill/references/contribution.md` 的核心贡献契约及 Motivation 规范核心章节；逐段核验 `src/scripts/paperspine5_web.py` 的 CLI 入口、`src/scripts/paperspine_public_client.py` 的 `web_launch`/`call` 及 MCP task snapshot、`src/scripts/manuscript_export.py` 的 Pandoc AST 与 DOCX/LaTeX 导出调用、`src/scripts/contribution_check.py` 的旧式贡献检查器。此为核心原文与关键运行入口的阅读核对，非逐行覆盖整个仓库；未执行端到端生产或渲染。
- **通读：** 已完成
- **知识页：** [PaperSpine](../../wiki/entities/paperspine.md)

## 官方原文

- [README 原文](https://github.com/WUBING2023/PaperSpine/blob/42312b8d9c76/README.md)：论文全流程与 Word/LaTeX/PDF 交付能力。
- [PaperSpine Skill](https://github.com/WUBING2023/PaperSpine/blob/42312b8d9c76/src/skill/SKILL.md)：preflight、材料绑定、Web 配置、同一 task ID 的论文生产与审阅。
- [Contribution-First 规范](https://github.com/WUBING2023/PaperSpine/blob/42312b8d9c76/src/skill/references/contribution.md)：一条可辩护贡献、证据充分性、Claim Boundary 与 Motivation 分离。
- [Motivation 写作规范](https://github.com/WUBING2023/PaperSpine/blob/42312b8d9c76/src/skill/references/motivation-thread-writing.md)：贡献确认后生成论文主线与章节映射。
- [Web Launcher 源码](https://github.com/WUBING2023/PaperSpine/blob/42312b8d9c76/src/scripts/paperspine5_web.py)、[贡献检查器](https://github.com/WUBING2023/PaperSpine/blob/42312b8d9c76/src/scripts/contribution_check.py)、[导出器](https://github.com/WUBING2023/PaperSpine/blob/42312b8d9c76/src/scripts/manuscript_export.py)。

## 可复核结论

PaperSpine 的 Contribution-First 并不是简单“先润色”：贡献类型、需要的证据、已有证据、缺失证据、Novelty/Significance 风险要先可解释；Motivation 解释为什么值得研究，而不是替代贡献。当前 V5 文档强调复用已确认的用户选择，不应把 V4 的旧文件检查器硬编码为 V5 每次工作的必经流程。

## 补充核验：17 步质量协议与代码运行边界

核心 [17 步生产协议](https://github.com/WUBING2023/PaperSpine/blob/42312b8d9c76/src/skill/references/paper-spine-production-protocol.md) 已全文阅读：1–3 建立同一任务绑定、材料和配置；4–8 选择科研方法、学习目标期刊、清查证据、确认贡献并开展获授权的分析；9–12 明确图表职责、阅读参考图并由用户保存选图；13–15 从单一语义文稿生成引文一致的源文件、DOCX/LaTeX/PDF；16–17 检查实际页面、独立审阅、打包并在同一任务内修订。

- `src/scripts/paperspine5_web.py` 的 `main` 将 Web 启动或 host 请求转交 `paperspine_public_client.py` 的 `web_launch` / `call`；Web 绑定同一 profile/task，配置和选择需要真实用户保存。
- `paperspine_public_client.py` 将 `host snapshot` 映射到 `paperspine_get_task`，`host wait` 到 `paperspine_wait_for_task_change`；二者不是代理自行执行科学研究的进程。
- `src/scripts/manuscript_export.py` 使用 Pandoc 从 Markdown 生成 JSON AST，再面向 DOCX 或 LaTeX 导出，PDF 另依赖 TeX 引擎；导出成功不代表学术、页面视觉或投稿质量已经过关。

**仍未验证：** 实机任务全链执行、多平台安装包、全部可选插件与最终视觉质量。此类测试与“已通读上述官方原文”是两类证据，不能互相替代。