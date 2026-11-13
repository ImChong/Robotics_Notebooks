## [2026-11-13] ingest | Embodied First Exam 论文协议、九套件与失败阶段深化

- 复用已入库的 arXiv:2611.12177 唯一实体 `wiki/entities/embodied-first-exam.md`，不重复建立同论文第二个节点。
- 根据论文 v1（正文 §1–5、附录 A–K 的既有通读记录，以及本次 Figure 3、§2–3、Appendix I–J 原文复核），补充三种控制路径、三类模式、九个原创套件与 Mermaid 对应数据流。
- 澄清 456 核心 Standard 33.6%、391 配对任务 Standard 30% vs Privileged 77%、391 受控消融 28% → 45% / 53% / 68%，避免跨分母误读。
- 补充阶段失败、Realtime 配对、harness 差异、真机演示边界；按 2026-11-13 官方 README 更新代码仓扩增至 36 benchmarks / 1,211 tasks 与论文 35/1,156 区分。
- 保留原先逐文件核对的官方源码时序图；资料归档更新 `sources/papers/`、`sources/repos/` 与 `sources/sites/`。
