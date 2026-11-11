# 2026-11-11 ingest | Ai2 ToyBlocks 原始研究深化与 Sim2Real 边界核实

- **意图：** 响应 ToyBlocks 单项目 ingest，发现 2026-11-09 已由 PR #3551 建立主节点，因而选择更新现有唯一实体、补充定量实验与真实源码执行链，避免重建同项目节点。
- **原文：** https://huggingface.co/blog/allenai/toyblocks （2026-11-06）；https://allenai.github.io/ToyBlocks/experiments.html；https://allenai.github.io/ToyBlocks/sim2real.html；https://github.com/allenai/ToyBlocks
- **开源结论：** 仿真/数据生成、模型训练/评测、官方模型与数据已公开；FR3 真机执行工具包与 system identification 指南截至本次核查仍标注 Coming soon，完整一体化 RL 训练尚待补充。
- **主要更新：** VLA 空间泛化的五种对照方法、智能体与世界模型定量结果、FR3 真机与仿真分母区分、代码核验的 datagen 与 eval 时序图。
- **主节点：** [ToyBlocks](../wiki/entities/toyblocks.md)
- **核验边界：** 通读官方发布原文与实验/Sim2Real 页面，核对运行入口代码；未执行 Isaac、GPU 训练或真机部署。
