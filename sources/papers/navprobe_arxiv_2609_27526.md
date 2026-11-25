# NavProbe — arXiv:2609.27526

- **标题：** NavProbe: Evidence-Grounded Reasoning with Active Memory Retrieval for Zero-Shot Navigation
- **类型：** paper
- **作者：** Jingyang Liu, Sujia Yao, Jiayuan Gu, Lan Xu
- **机构：** 上海科技大学（ShanghaiTech University）
- **日期：** 2026-09-23；v2 2026-10-03
- **论文：** https://arxiv.org/abs/2609.27526
- **PDF：** https://arxiv.org/pdf/2609.27526v2
- **HTML：** https://arxiv.org/html/2609.27526v2
- **代码：** https://github.com/liujy25/NavProbe
- **核对范围：** v2 HTML 正文 I–V、公式 (1)–(4)、图 1–3、表 I–IV、参考文献；未独立复现实验。
- **通读：** 已完成

## 核心摘录（归纳）

1. §III-B：子目标可随证据改变；任务状态、原始图记录和语义索引分开存储。
2. §III-C：需要判断时才请求具体实体字段，证据可推翻旧结论；EKM 将可复用结论写回。
3. §III-D：FSS 提供路点，技能策略选择动作，执行补充新的历史记录。
4. §IV：三个基准分别采用分层抽样的 100 个 episode，三次运行取均值；R2R-CE SR/SPL 71.7/55.8，RxR-CE 55.3/38.6，HM3D-v2 79.3/43.8。消融支持任务状态、选择性检索与巩固，不等于证明实时性。
5. §IV-E：Galaxea R1Pro 使用 ZED2、MID-360、Voxel-SLAM 和 Nav2；真机展示为定性结果。

对 wiki 的映射：[NavProbe](../../wiki/entities/paper-agentic-2609-27526.md)。

源码归档：[NavProbe repository](../repos/navprobe.md)。
