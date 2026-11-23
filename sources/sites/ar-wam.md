# AR-WAM 官方项目主页

> 一手项目归档，论文身份复用 [paper-ar-wam](../../wiki/entities/paper-ar-wam.md)，不另建项目节点。

- **标题：** AR-WAM: A Visual-Conditioned Agent-Ready World Action Model for Robotic Manipulation
- **类型：** site
- **项目主页：** https://ar-wam.github.io/
- **论文：** https://arxiv.org/abs/2609.23578
- **代码：** https://github.com/AR-WAM/AR-WAM
- **数据集：** https://modelscope.cn/datasets/Jasonggg15/AR-WAM-RoboTwin-Dataset
- **核对日期：** 2026-11-22
- **核对范围：** 官网标题/作者、Code/Dataset 链接、Abstract、Method Overview、Visual Prompt、Reasoning+Foresight、Agent-Policy Loop、RoboTwin 2.0/RMBench/Real-World/Efficiency 与各实验表。
- **通读：** 已完成

## 原始页面要点

- 官网公开 **Code（GitHub）和 Dataset（ModelScope）**；模型训练源码可用，但对 Agent 真机部署的完整端到端脚本与 AR-WAM 权重需另行核查，不能把论文 demo 当可执行代码。
- **方法定位：** 外部大模型将语言理解和复杂任务拆解成目标框+动作 token，0.5B 策略内部无语言编码器；冻结视觉骨干，推理时通过共享 Transformer 产生 reasoning 输出与动作/未来潜表征。
- **评测口径：** RoboTwin 2.0 clean 87.2%、randomized 84.2%；RMBench M(n) AR-Q 63.5%、AR-K 52.5%；Astribot S1 真机 71.7% vs π₀.₅ 35.0%；4090 模型端 14.09 ms/chunk。
- **源码入口：** [官方仓库归档](../repos/ar-wam.md)；原文详细条件/局限见[论文归档](../papers/ar-wam-arxiv-2609-23578.md)。

## 对 wiki 的映射

- [唯一论文详情](../../wiki/entities/paper-ar-wam.md)
