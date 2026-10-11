# UltraText Bench（LINs-lab/UltraText_Bench）

> 来源归档（repo）

- 类型：repo / bilingual image-text benchmark
- 仓库：https://github.com/LINs-lab/UltraText_Bench
- 项目页：https://lins-lab.github.io/UltraText_Bench/
- 论文：https://arxiv.org/abs/2610.09823
- 检查日期：2026-10-11
- README 列有中英文 prompt 与结构化参考、Qwen-Image/Z-Image 生成 runner、Q-Judger 评估客户端、分数聚合、coverage 报告与回归测试。
- 离线验证入口：`python scripts/rebuild_audit_snapshots.py --check`；`python -m unittest discover -s tests -v`。
- 完整图像档案和 Q-Judger checkpoint 未打包；图像生成需要模型权重/CUDA，评估需要另行运行兼容 judge service。
- Wiki：[UltraText Bench](../../wiki/entities/paper-ultratext-bench.md)

## 开源边界

仓库具备 benchmark 数据和实际评测客户端，但开放客户端并不等于完整排行榜一键复现；应一并固定生成器、judge checkpoint、配置和 coverage。
