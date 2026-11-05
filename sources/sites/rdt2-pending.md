# RDT2 项目页（待完整核验）

- **项目：** RDT2: Exploring the Scaling Limit of UMI Data Towards Zero-Shot Cross-Embodiment Generalization（RDT2：扩展 UMI 数据以实现零样本跨具身泛化）
- **项目页：** <https://rdt-robotics.github.io/rdt2/>
- **论文：** <https://arxiv.org/abs/2602.03310>
- **官方代码：** <https://github.com/thu-ml/RDT2>
- **权重：** <https://huggingface.co/collections/robotics-diffusion-transformer/rdt-2>
- **核查日期：** 2026-11-05

## 已核对的项目页线索

RDT2 是 RDT-1B 的后续工作。官方项目页描述使用超过 10,000 小时 UMI 数据，先通过残差向量量化训练 VLA，继而训练 flow-matching 动作专家，并探索单步蒸馏。项目页指出水、热接触及五指灵巧操作受采集硬件限制。官方代码仓列出 RDT2-VQ 和 RDT2-FM 代码及权重；项目页将 FM-Post 和 UltraFast 标为待发布，具体发布状态应继续核查，不应将其与已发布模型混同。

## 待完成

- 通读论文全文及附录，核对实验口径、局限与项目页差异。
- 阅读官方训练及推理源码入口，核实当前版本的代码、权重和数据开放范围。
- 建立独立 RDT2 实体，复用已有 RDT-1B 节点并交叉链接；补项目流程图和基于源码的运行时序图。
- 运行仓库质量检查。当前为资料归档线索，未完成 ingest。
