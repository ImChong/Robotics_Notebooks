# LongLive-2.0 官方项目页归档（NVIDIA）

- **类型：** site
- **标题：** LongLive-2.0: An NVFP4 Parallel Infrastructure for Long Video Generation
- **官方网站：** <https://nvlabs.github.io/LongLive/LongLive2/>
- **使用文档：** <https://nvlabs.github.io/LongLive/LongLive2/docs/>
- **论文：** <https://arxiv.org/abs/2605.18739>
- **代码：** <https://github.com/NVlabs/LongLive/tree/main/LongLive2.0>
- **模型权重：** <https://huggingface.co/Efficient-Large-Model/LongLive-2.0-5B>、<https://huggingface.co/Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S2>、<https://huggingface.co/Efficient-Large-Model/LongLive-2.0-5B-NVFP4-S4>
- **开源状态：** **已开源**（代码、训练/推理入口、模型下载指引；第三方预训练权重依各自许可）
- **核对范围：** 官方页面 Overview、Motivation、Technical Walkthrough、Key Results、BibTeX，官方 Docs 安装、训练、推理及 Troubleshooting；对照 GitHub 主 README。
- **通读：** 已完成
- **核验日期：** 2026-11-08
- **沉淀到 wiki：** [LongLive-2.0 主实体](../../wiki/entities/paper-longlive-2-nvfp4-long-video-generation.md)

## 关键事实

- 2026-05-13 官方 GitHub README 公告发布 LongLive 2.0；并非 2026-11-08 新公开。
- 官方项目页写明训练侧 Balanced SP + NVFP4，推理侧 W4A4 NVFP4、KV Cache 量化和 async VAE。
- 官网给出 45.7 FPS、19.4 GB 峰值、64s 视频 36.3s 端到端延时；应同时参考论文 Table 3 的 GPU 计量与异步解码条件。
- 官方 NVFP4 环境相对 BF16 更挑版本：Python 3.12.12、PyTorch 2.10.0+cu128、CUDA 12.8、FlashAttention 2.8.3，Blackwell 目标需编译 CUDA 扩展。

## 相关归档

- [论文](../papers/longlive2_nvfp4_arxiv_2605_18739.md)
- [源码](../repos/nvlabs-longlive2.md)
