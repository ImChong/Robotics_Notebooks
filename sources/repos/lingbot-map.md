# LingBot-Map

- **标题**: Geometric Context Transformer for Streaming 3D Reconstruction
- **链接**: [https://github.com/Robbyant/lingbot-map](https://github.com/Robbyant/lingbot-map)（组织名大小写不敏感；GitHub 展示为 `robbyant/lingbot-map`）
- **类型**: repo
- **作者**: Chen, Lin-Zhuo, et al. (2026)
- **摘要**: LingBot-Map 是面向**流式 3D 重建**（从连续视频在线估计位姿与稠密几何）的**前馈式 3D 基础模型**。核心为 **Geometric Context Transformer**：在单一流式框架内用 **anchor context、pose-reference window、trajectory memory** 统一坐标接地、稠密几何线索与长程漂移校正；推理侧推荐 **FlashInfer** 的 **Paged KV cache**，官方 README 报告在 **518×378** 分辨率约 **20 FPS**，并给出 **>10,000 帧** 长序列与 **窗口化（windowed）**、**keyframe 间隔** 等工程策略。

## URL 勘误（常见误链）

- 公开资料中偶见 `https://github.com/byant/Lingbot-map`：**该路径在 GitHub 上返回 404**（截至 2026-05-17）。**官方开源入口为** [Robbyant/lingbot-map](https://github.com/Robbyant/lingbot-map)。

## 核心要点

1. **流式推理**: 前馈因果模型 + KV 缓存，避免传统 BA 式全局迭代；无 test-time training。
2. **长序列**: **Paged KV cache**；超过训练 RoPE 覆盖的帧数时需 **keyframe 抽稀** 或 **windowed** 滑动窗口（详见官方 README 的 demo / batch 说明）。
3. **权重与数据**: Hugging Face / ModelScope 发布 **`lingbot-map-long`**（偏长序列/大场景）、**`lingbot-map`**（均衡）、**`lingbot-map-stage1`**（可载入 VGGT 做双向等实验）；演示序列数据集 **`robbyant/lingbot-map-demo`**。
4. **技术栈**: DINOv2 系 ViT 特征；与 VGGT 系交替 **Frame Attention** 与 **GCA**；可选 **Viser** 交互、`demo_render/batch_demo.py` 离线渲染（依赖 **Kaolin**、CUDA 扩展与 ffmpeg 等）。
5. **论文**: arXiv:2604.14141；项目页 [technology.robbyant.com/lingbot-map](https://technology.robbyant.com/lingbot-map)。

## 为什么值得保留

- 把「SLAM 式三类空间记忆」收进**可学习注意力**，为机器人/AR 的**在线几何感知**提供可复现开源栈与权重。
- 工程 README 对 **RoPE 训练窗（约 320 views）**、**keyframe_interval**、**overlap_keyframes** 等有可操作说明，利于 sim2real 与 VLA 几何先验集成讨论。

## 对 wiki 的映射

- [wiki/methods/lingbot-map.md](../../wiki/methods/lingbot-map.md): 方法页（论文机制 + 仓库工程注意）
- [wiki/methods/vla.md](../../wiki/methods/vla.md): 作为可选几何先验/环境表示参照

---
- **录入日期**: 2026-04-27（初录）；**2026-05-17** 增补论文/站点索引、误链勘误与 README 工程要点
