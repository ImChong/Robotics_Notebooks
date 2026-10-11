# UltraText Bench: A Comprehensive Bilingual Benchmark for Evaluating Visual Text Rendering in Image Generation

> 来源归档（paper）

- 类型：paper / benchmark / image-generation evaluation
- arXiv：https://arxiv.org/abs/2610.09823v1
- 作者：Deyuan Liu、Yihao Hu、Jingxuan Zhang 等
- 机构：西湖大学、蚂蚁集团、浙江大学、上海创新研究院、香港科技大学等
- 项目页：https://lins-lab.github.io/UltraText_Bench/
- 代码与数据：https://github.com/LINs-lab/UltraText_Bench
- 入库日期：2026-10-11

## 摘录与 wiki 映射

- 基准有 432 个 prompt、24 类场景、3 档难度、2,926 个文本区域；中英文各 216 个，每条样本 4–12 个区域。
- 区域参考包括目标字符串、位置和视觉属性；Q-Judger 结合完整参考评分，汇总为 fidelity、clarity、spatial、scene 四维。
- 论文比较 24 组模型配置，指出可读性与内容忠实度并非同一指标，难度上升会暴露性能下降。
- 仓库含 prompt 数据、生成 runner、judge 客户端和测试；完整生成图档与 Q-Judger checkpoint 未随仓库提供。
- 对 wiki 的映射：[UltraText Bench 实体](../../wiki/entities/paper-ultratext-bench.md)；[项目页](../sites/ultratext-bench.md)；[代码归档](../repos/ultratext-bench.md)。
