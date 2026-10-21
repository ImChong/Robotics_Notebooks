# MemT-1.0 官方项目总览

> 来源归档（site）。MemTensor 当前提供的官方总览入口是 GitHub 仓库 README。

- **标题：** MemT-1.0: Towards Memory-Native Evolution
- **类型：** site（官方项目总览）
- **链接：** <https://github.com/MemTensor/MemT-1.0>
- **论文：** [arXiv:2610.26825](../papers/memt_1_0_arxiv_2610_26825.md)
- **代码：** [MemTensor/MemT-1.0](../repos/memtensor-memt-1-0.md)
- **模型发行：** [Hugging Face](https://huggingface.co/collections/MemTensor/memt-1) · [ModelScope](https://www.modelscope.cn/collections/MemTensor/MemT-10)
- **入库日期：** 2026-10-20
- **一句话说明：** MemTensor 发布的原生记忆基础模型项目，介绍架构、Nano/Flash 模型族、评测、agentic coding 应用及推理入口。

## 项目页摘录

- **记忆架构：** Native Memory Block 与 backbone attention 并行工作，包含动态、多通道的 Local Memory Block 和含可训练投影及门控的 Hyper Memory Block。
- **评测口径：** README 称 Nano 与 Flash 在被评估的参数化记忆方法中排名前二，同时说明强文本记忆系统整体仍领先，长历史细粒度证据检索尤其如此。
- **应用：** 项目展示将程序性经验写入原生记忆并供长程 coding agent 在执行时调用；这不是机器人闭环控制或真机验证。
- **发布范围：** 仓库软件、论文材料和模型权重各自适用不同许可，使用者应按目标文件对应的 LICENSE / model card 确认。

## 对 wiki 的映射

- [MemT-1.0 实体页](../../wiki/entities/paper-memt-1-0-memory-native-evolution.md)
- [arXiv 技术报告摘录](../papers/memt_1_0_arxiv_2610_26825.md)
- [官方实现仓库归档](../repos/memtensor-memt-1-0.md)
