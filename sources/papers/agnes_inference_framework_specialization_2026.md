# Inference Framework Specialization: Methodology and Interim Results · Agnes 推理框架专用化技术报告

- **类型：** 技术报告
- **机构：** Agnes AI 实验室（Agnes AI Labs）
- **版本：** 英文修订版 v2，2026-11-13，32 页
- **官方发布仓库：** https://github.com/AgnesAI-Labs/Inference-Framework-Specialization
- **PDF：** https://github.com/AgnesAI-Labs/Inference-Framework-Specialization/blob/95296c62086c807e4298507009b67995fb5eadf7/PAPER.pdf
- **官方图解：** https://github.com/AgnesAI-Labs/Inference-Framework-Specialization/blob/95296c62086c807e4298507009b67995fb5eadf7/REPORT.md
- **核对范围：** 通读 PDF 第 1–32 页，包括 §1–3 的部署前提与指标、§4–7 六个组件研究、§8 集成结果、§9 验收缺口、图 1–22 与表中原值；核对固定提交的公开文件清单，按原值重算主要比例。
- **通读：** 已完成

## 原始证据索引

| 位置 | 内容 |
|---|---|
| §1–3 | Blackwell、PCIe-only、受限显存；混合注意力示例含 16 层全注意力、48 层 GDN，分别维护随历史增长的 KV 与循环状态。区分本地读取、跨卡规约、主机 KV 恢复和排队。 |
| §4、图 7–8 | FP4 attention 独立研究：512K 预填充 P50 为 231.9456→75.1677 秒；1K–2K 约慢 3%；固定 200 题 LongBench v2 为 64.50%→63.50%。 |
| §5–7、图 9–16 | 权重预取、算子选择、微批重叠、NVFP4 KV 和 BOLE；单测有各自资源配置、精度和延迟边界。 |
| §8、图 17–21 | 整体配置的 GPU KV 为 530,400→927,000 token；512K 冷预填充为 231.71→65.54 秒，汇总未注明延迟统计量；活跃请求上限 8→16。 |
| §9、图 22 | 完整质量、真实负载与长期稳定性仍待验收；跨硬件、跨模型需要独立验证；组件收益不可相加。 |

固定提交公开了 PDF、README、图解和四张图，没有发行该内部推理实现、部署脚本、完整运行配置或原始性能分布。图解不是额外试验。未运行 Blackwell 基准。

## 对 wiki 的映射

- [Agnes 推理框架专用化](../../wiki/entities/paper-agnes-inference-framework-specialization.md)
