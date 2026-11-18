# Code World Model 2.0 / Invoker-Render 官方主页

> 一手项目页归档（2026-11-18）

- **标题：** Invoker-Render · Code World Model 2.0: Towards Infinite Controllable Video Generation
- **项目页：** https://www.inclusion-ai.org/CodeWorldModel/
- **代码：** https://github.com/inclusionAI/CodeWorldModel （[仓库归档](../repos/inclusionai-codeworldmodel.md)）
- **模型：** https://huggingface.co/inclusionAI/invoker-render-1.0 （[模型归档](../models/invoker-render-1-0.md)）
- **类型：** site（作者官方演示、控制方式和优化基础设施）
- **核对范围：** 通读首页代理结构/首帧外观、相机变化、世界动态、Proxy 对应关系、细节粒度、交互赛车和 Infra 图文；核对 GitHub/HF 入口与“arXiv coming soon”状态。未从网页下载或逐帧重新测量所有视频。
- **通读：** 已完成

## 源码和权重开放核查

项目页直接指向 GitHub 推理代码和 HF 权重，**已开放推理实现与权重**；未核实训练数据或训练程序开源，GitHub README 将两者列为未发布。

## 官方展现的可控性与性能

- **结构与外观分工：** Proxy 决定场景布局、运动、相机路线；参考首帧给出外观，视频模型生成剩余视觉内容。
- **控制实验：** 相机路线 A/B、交叉路口交通控制、桥梁开合、几何体/框/骨架多粒度 Proxy 及实时交互赛车。
- **流水线指标：** 8 GPU Ulysses8、四次 DiT 去噪；官方报告 930 ms/块、18.0 FPS、相对 1770 ms baseline 为 1.82×。
- **重叠策略：** 下一块 Proxy 使用轻量 TAE 编码、当前块解码 RGB、CPU 异步发送；不能将优化流水线的时延移用于其他配置。
- **可信度说明：** 这些是官方报告及演示，尚无本次独立复测；视频逼真度不能作为物理正确性或机器人操控质量证据。

## Wiki 映射

- [Code World Model 2.0 / Invoker-Render](../../wiki/entities/code-world-model-2-invoker-render.md)
