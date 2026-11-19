# Alloy: A Multi-Material Physics Engine for Robot Learning 官方源码核对

- **类型：** 官方源码仓
- **代码：** https://github.com/argonrobotics-ai/alloy
- **论文：** https://arxiv.org/abs/2611.16312
- **公开运行入口：** `python examples/api/01_drop_box.py`
- **相关模块：** `alloy.Scene、alloy.build 与 Simulation`
- **运行产物：** state、health、render
- **核对范围：** 官方 README、该仓库公开入口脚本与关键模块的静态核查；外部数据和模型依赖未执行。
- **通读：** 已完成
- **wiki：** [论文独立详情](../../wiki/entities/paper-alloy-multimaterial-2611-16312.md)
- **原文：** [论文来源](../papers/alloy-multimaterial-2611-16312.md)

各类材料的参数、求解步长和物理精度边界不同；健康度 OK 只说明求解器运行稳定，不能代替实物物性校准。

## 0.1.7 版本复核（2026-11-19）

- **发布链接：** https://github.com/argonrobotics-ai/alloy/releases/tag/v0.1.7
- **发布时间：** 2026-11-18 03:48:59 UTC，官方 release v0.1.7。
- **安装：** `pip install alloy-engine==0.1.7`；Python 3.11/3.12。
- **实质变化：** 液体表面张力选项、ultra 粒子密度、渲染白平衡、内存泄漏修复和 CPU 估时调整。
- **局限：** 官方仍列出未解决的数值稳定与重置可复现问题；此处未实际运行。
