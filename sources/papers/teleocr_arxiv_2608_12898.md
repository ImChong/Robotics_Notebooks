# TeleOCR: Navigating Document Parsing Across Digital and Camera-Captured Documents（arXiv:2608.12898）

> 来源归档（ingest）

- **标题：** TeleOCR: Navigating Document Parsing Across Digital and Camera-Captured Documents
- **类型：** paper / vlm / document-parsing / ocr
- **arXiv abs：** <https://arxiv.org/abs/2608.12898>
- **PDF：** <https://arxiv.org/pdf/2608.12898>
- **代码：** <https://github.com/caipeng328/TeleOCR> — 归档见 [`sources/repos/teleocr.md`](../repos/teleocr.md)
- **权重：** <https://huggingface.co/StarDoc-AI/TeleOCR>
- **在线体验：** <https://www.teleai.com.cn/docparse/documentParsing> — 归档见 [`sources/sites/teleai-docparse.md`](../sites/teleai-docparse.md)
- **机构：** 中国电信人工智能（TeleAI）/ StarDoc-AI 团队 — Peng Cai、Zhaofan Zou、Shifa Liu 等
- **入库日期：** 2026-09-30
- **一句话说明：** **~1.2B** 专用文档 VLM，统一 **数字 PDF** 与 **手机拍摄畸变文档** 的版面+内容解析；OmniDocBench v1.6 **Overall 96.87**、Wild-OmniDocBench **88.53**、PureDocBench 多轨领先；**GitHub + HF 已开源**，TeleAI 提供在线 demo。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 仓库 | <https://github.com/caipeng328/TeleOCR> | `infer.py` 批量推理、vLLM 后端、`LAYOUT_MODE` |
| HF | <https://huggingface.co/StarDoc-AI/TeleOCR> | 权重 + Transformers 单页 snippet |
| 前身权重 | <https://huggingface.co/StarDoc-AI/NaviDC-OCR> | 2026-09-10 起品牌统一为 TeleOCR |
| 在线 demo | <https://www.teleai.com.cn/docparse/documentParsing> | 免本地部署试解析 |
| 评测 | [OmniDocBench v1.6](https://github.com/opendatalab/OmniDocBench) | 主榜 Docker 评测口径 |

## 摘要级要点

- **问题：** 解耦式 VLM 管线依赖精确版面，拍摄畸变易级联出错；端到端 VLM 在高分辨率下易冗余生成与结构幻觉。
- **方法：** **变形感知学习** + **CGDP 自适应采样** + **内容–结构解耦**（公式文法、表格 OTSL/HTML）；**MCV 伪标** + **几何合成** + **图到图自验证** + **四阶段渐进训练**（对齐 → 几何解析 → 解耦 → RL）。
- **规模：** 约 **1.2B** 参数；基于 Qwen2.5-VL / Qwen3、MinerU 生态。
- **命名：** 2026-09-10 **NaviDC-OCR → TeleOCR**，后续迭代统一 TeleOCR 品牌。

## 核心摘录（面向 wiki 编译）

### 1) OmniDocBench v1.6（Specialized VLMs，节选）

| Method | Params | Overall ↑ | Text Edit ↓ | Table TEDS ↑ |
|--------|--------|-----------|-------------|--------------|
| **TeleOCR** | 1.2B | **96.87** | 0.027 | **97.05** |
| OvisOCR2 | 0.8B | 96.58 | **0.025** | 94.76 |
| MinerU2.5-Pro | 1.2B | 95.75 | 0.036 | 93.42 |

### 2) Wild-OmniDocBench（Decoupled VLMs）

- TeleOCR **88.53** Overall，领先同参 MinerU2.5-Pro（87.33）与 PaddleOCR-VL-1.6（87.36）。

### 3) Dr.DocBench Challenge（EMNLP 2026 赛道，自报）

- TeleOCR **67.96** overall，高于 MinerU 2.5 pro（62.26）与 PaddleOCRvl 1.6（55.11）。

### 4) 拍摄畸变

- DocUNet / DIR300 可视化：**无需去畸变预处理** 即可做多点 layout 分割与内容解析。

## 对 wiki 的映射

- 新建：[paper-teleocr](../../wiki/entities/paper-teleocr.md)
- 交叉：[auto-labeling-pipelines](../../wiki/methods/auto-labeling-pipelines.md)（手册/SOP 结构化上游）、[llm-wiki-karpathy](../../wiki/references/llm-wiki-karpathy.md)（PDF 资料 ingest 辅助）

## 当前提炼状态

- [x] arXiv + GitHub/HF/TeleAI 页核查（**已开源**）
- [ ] 训练数据引擎细节与完整文档解析管线以仓库后续更新为准
