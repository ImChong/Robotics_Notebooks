## [2026-11-10] ingest | LingBot-Touch | 异构触觉 JEPA 表征与官方推理实现

- 意图：为 Robbyant LingBot-Touch 建立唯一项目实体，关联 LingBot 家族。
- 已核验：官网 4,983h / 三模态、26 真机 + 12 仿真任务、GitHub 推理入口/输入契约、Tiny/Small 冻结 encoder 和 Apache-2.0。
- 开源边界：当前发行提供推理链、样例及权重，不能等同于完整预训练开源。
- 复核完成：通过临时 GitHub Actions 按 SHA-256 校验并逐页阅读官方 PDF **27 页**（正文 §1–6、Appendix A–C、References）；技术报告和评测表已进入 sources/papers 与主实体。
- 关键纠正：论文 Table 2 的 26 项真机结果基于 **4K 预训练 Tiny**，§4 frozen probing 是 **5K Small**，§5.4 单独对比 **1K–5K Tiny**；多源预训练在固定总时长下存在空间精度 trade-off。
- 节点健康度：新实体与 Robbyant 公司总览均补齐原文核验锚点，目标按报告脚本验证 **3/3**；未做本地 GPU/真机数值复现。
- 主节点：[LingBot-Touch](../wiki/entities/paper-lingbot-touch.md)；[论文记录](../sources/papers/lingbot-touch-technical-report.md)。
