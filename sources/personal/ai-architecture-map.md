# AI 架构地图（维护者整理的函数族 taxonomy）

- **类型**：`personal`（对话/选型整理，非正式出版物）
- **日期**：2026-09-21
- **用途**：为 [AI 架构地图](../../wiki/overview/ai-architecture-map.md) 提供可追溯的编译来源；正文以 wiki 页为准，本文件只固定 **六支函数族** 的阅读坐标。

## 一句话说明

按 **数据几何与任务接口** 把神经网络分成：前馈、空间、序列、生成、图、机器人决策六支，而不是按发表年代堆砌模型名。

## 六支坐标

```text
神经网络 / AI 模型
├─ 1. 基础前馈网络          MLP / Feedforward NN · MoE
├─ 2. 图像 / 空间结构       CNN · ResNet · U-Net · ViT
├─ 3. 序列 / 时间建模       RNN · LSTM · GRU · TCN · Transformer · SSM · Mamba
├─ 4. 生成模型              Autoencoder · VAE · GAN · Diffusion · DiT
├─ 5. 图结构                GNN
└─ 6. 机器人 / 决策模型     MLP Policy · RNN Policy · Transformer Policy · Diffusion Policy · VLA
```

## 对 wiki 的映射

- 总图：[`wiki/overview/ai-architecture-map.md`](../../wiki/overview/ai-architecture-map.md)
- 一手论文簇：[`sources/papers/ai_architecture_foundations.md`](../papers/ai_architecture_foundations.md)
