# RAEv2 官方项目主页

- **标题：** Improved Baselines with Representation Autoencoders / 改进表征自编码器基线
- **类型：** site / paper project
- **项目：** https://raev2.github.io/
- **论文：** https://arxiv.org/abs/2605.18324v2
- **代码：** https://github.com/nanovisionx/RAEv2
- **模型：** https://huggingface.co/nyu-visionx/RAEv2-models
- **数据：** https://huggingface.co/datasets/nanovisionx/RAEv2-data
- **核查日期：** 2026-11-19
- **核对范围：** 首页资源、Method 2.1–2.3、Results、T2I 与导航世界模型页段落；以项目页 Code 实际指向的官方仓库核查开放情况。
- **通读：** 已完成

## 开放程度

**已开源：** 项目页提供官方 GitHub 训练/评测实现，以及 Hugging Face 权重、数据资源入口；不是“即将开源”。实现与实验范围见 [仓库归档](../repos/raev2.md)。本次仅核验公开内容、未执行训练复现。

## 核心内容

- K 层 feature aggregation 调整重建–生成权衡；
- 同一表征作 RAE 输入与 REPA 蒸馏目标；
- REPA head 支持内部自引导，不另训弱扩散模型；
- ImageNet-256、T2I、RECON 导航帧生成三种评测边界须分开。

## 来源与映射

- [论文通读记录](../papers/raev2_improved_baselines_arxiv_2605_18324.md)
- [独立论文详情](../../wiki/entities/paper-raev2-improved-representation-autoencoders.md)
- [官方项目主页](https://raev2.github.io/)
