# PaperSpine5（官方上游仓库）

- **类型：** repo / 论文写作系统
- **原始仓库：** https://github.com/WUBING2023/PaperSpine
- **上游身份：** GitHub REST `fork: false`；非 michae1liu/paperspine Fork
- **核验版本：** `42312b8d9c76`（main，2026-10-14）
- **入库日期：** 2026-11-11
- **许可：** MIT
- **发布状态：** 官方 README 对应 PaperSpine5，链接 v0.4.0-alpha.3；仍为 Alpha
- **开源状态：** 已发布 SKILL、Python 脚本、Web 接入层及安装包，部分功能需本地依赖（如 Pandoc/LaTeX）与配置
- **核对范围：** 已通读官方 `README.md`；检查 `src/skill/SKILL.md` 的配置与任务边界、`src/skill/references/contribution.md` 的 Contribution-First 规则、`src/skill/references/motivation-thread-writing.md` 的 Motivation 机制；检查 `src/scripts/paperspine5_web.py`、`src/scripts/contribution_check.py` 与 `src/scripts/manuscript_export.py` 的入口代码局部。全量协议与运行代码尚未逐行通读，未实机运行。
- **通读：** 未完成
- **知识页：** [PaperSpine](../../wiki/entities/paperspine.md)

## 官方原文

- [README 原文](https://github.com/WUBING2023/PaperSpine/blob/42312b8d9c76/README.md)：论文全流程与 Word/LaTeX/PDF 交付能力。
- [PaperSpine Skill](https://github.com/WUBING2023/PaperSpine/blob/42312b8d9c76/src/skill/SKILL.md)：preflight、材料绑定、Web 配置、同一 task ID 的论文生产与审阅。
- [Contribution-First 规范](https://github.com/WUBING2023/PaperSpine/blob/42312b8d9c76/src/skill/references/contribution.md)：一条可辩护贡献、证据充分性、Claim Boundary 与 Motivation 分离。
- [Motivation 写作规范](https://github.com/WUBING2023/PaperSpine/blob/42312b8d9c76/src/skill/references/motivation-thread-writing.md)：贡献确认后生成论文主线与章节映射。
- [Web Launcher 源码](https://github.com/WUBING2023/PaperSpine/blob/42312b8d9c76/src/scripts/paperspine5_web.py)、[贡献检查器](https://github.com/WUBING2023/PaperSpine/blob/42312b8d9c76/src/scripts/contribution_check.py)、[导出器](https://github.com/WUBING2023/PaperSpine/blob/42312b8d9c76/src/scripts/manuscript_export.py)。

## 可复核结论

PaperSpine 的 Contribution-First 并不是简单“先润色”：贡献类型、需要的证据、已有证据、缺失证据、Novelty/Significance 风险要先可解释；Motivation 解释为什么值得研究，而不是替代贡献。当前 V5 文档强调复用已确认的用户选择，不应把 V4 的旧文件检查器硬编码为 V5 每次工作的必经流程。

## 待复核

V5 17 步生产协议、各工具在运行时的完整调用关系及多平台安装包未做端到端验证，因此 wiki 暂保留草稿状态。