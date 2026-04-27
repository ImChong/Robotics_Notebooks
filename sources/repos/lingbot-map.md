# LingBot-Map

- **标题**: Geometric Context Transformer for Streaming 3D Reconstruction
- **链接**: [https://github.com/Robbyant/lingbot-map](https://github.com/Robbyant/lingbot-map)
- **类型**: repo
- **作者**: Chen, Lin-Zhuo, et al. (2026)
- **摘要**: LingBot-Map 是一个专为流式 3D 重建（Streaming 3D Reconstruction）设计的前馈式 3D 基础模型。旨在解决从连续视频流中高效、稳定地构建 3D 场景的问题。

## 核心要点

1. **流式推理 (Streaming Inference)**: 采用前馈架构，无需全局优化即可实时处理视频帧（约 20 FPS）。
2. **长序列支持**: 通过 Paged KV Cache 和窗口化推理技术，支持处理超长视频序列（>10,000 帧）。
3. **漂移校正 (Drift Correction)**: 在单一框架内集成了长程漂移校正机制。
4. **技术架构**: 核心为 Geometric Context Transformer (GCT)，构建于 VGGT 和 DINOv2 之上。
5. **高效性**: 支持 FlashInfer 加速，利用 NVIDIA Kaolin 进行几何处理。

## 为什么值得保留

- 代表了 3D 基础模型在流式感知和长序列重建方面的最新进展。
- 解决了 SLAM/重建中常见的长程漂移和大规模场景处理问题。
- 技术栈涵盖了当前主流的 3D 视觉与 Transformer 加速技术。

## 对 wiki 的映射

- `wiki/methods/lingbot-map.md`: 创建详细的方法说明页。
- `wiki/tasks/3d-reconstruction.md`: (待检查是否存在) 补充作为流式重建的代表方法。
- `wiki/concepts/slam.md`: 作为漂移校正和流式感知的参考。

---
- **录入日期**: 2026-04-27
