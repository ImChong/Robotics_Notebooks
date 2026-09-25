# Diffusion-based World Models — Living Survey（GitHub）

> 来源归档

- **标题：** Diffusion-based World Models: A Survey — GitHub Living Survey
- **类型：** repo / survey / awesome-list / world-models / diffusion
- **链接：** <https://github.com/energy588/Diffusion-based-World-Models>
- **论文：** [diffusion_wm_survey_preprints_202609_1022.md](../papers/diffusion_wm_survey_preprints_202609_1022.md)（DOI [10.20944/preprints202609.1022.v1](https://doi.org/10.20944/preprints202609.1022.v1)）
- **预印本：** <https://www.preprints.org/manuscript/202609.1022>
- **许可证：** 仓库 **未在 API 返回 license 字段**（截至 2026-09-25）；以 GitHub 页面为准
- **入库日期：** 2026-09-25
- **一句话说明：** **Project-page 风格** 配套仓：360+ 论文年表、`papers/` 三分域清单、数据集/benchmark 索引、原版与 premium 视觉图；欢迎 PR 补论文与链接。
- **开源状态：** **已开源（策展 / 文档型）** — 非单一 WM 训练框架

---

## 目录结构（README）

```
Diffusion-based-World-Models/
├── README.md              # 主论文表（按年）+ 视觉 banner
├── CONTRIBUTING.md        # 贡献规范
├── assets/                # taxonomy / galaxy / dataset 图
├── papers/
│   ├── autonomous-driving.md
│   ├── embodied-intelligence.md
│   ├── general-worlds.md
│   └── datasets.md
└── resources/
    ├── awesome-world-models.md
    ├── benchmarks.md
    └── survey-notes.md
```

## 使用方式

| 目标 | 入口 |
|------|------|
| 按年浏览代表论文 | `README.md` 年目录（2016–2026） |
| 分域深读 | `papers/autonomous-driving.md` 等 |
| 数据集 / benchmark | `papers/datasets.md`、`resources/benchmarks.md` |
| 贡献新论文 | `CONTRIBUTING.md` → PR |

## survey-notes 核心摘录

- **Core message：** Diffusion-WM 是在多模态条件下建模未来世界动态的 **生成基底**。
- **Strengths：** 高保真、多模态条件、灵活可控、不确定性建模。
- **Challenges：** 算力、长程一致、因果、验证与评测、实时交互。

## 边界说明

- **这是 Living Survey 维护仓**，不是某一篇 diffusion WM 论文的官方 `train.py`。
- 复现 **具体系统**（Cosmos、Ctrl-World、NWM 等）须跳转 README 内各 **Code** 链。
- Preprints 全文下载在部分环境受限；综述细节以 PDF + 本仓 `resources/survey-notes.md` 为辅。

## 对 wiki 的映射

- 实体页 [`paper-diffusion-based-world-models-survey.md`](../../wiki/entities/paper-diffusion-based-world-models-survey.md) — 「源码运行时序图」= 策展 PR 工作流
