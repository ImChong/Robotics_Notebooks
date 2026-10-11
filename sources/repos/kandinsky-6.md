# Kandinsky 6.0 Video（kandinskylab/kandinsky-6）

> 来源归档（repo）

- 类型：repo / synchronized video-audio generation
- 仓库：https://github.com/kandinskylab/kandinsky-6
- 项目页：https://kandinskylab.ai/models/video/
- 论文：https://arxiv.org/abs/2610.05608
- 许可证：MIT（官方仓库/报告声明）
- 检查日期：2026-10-11
- 运行入口：README 要求 NVIDIA GPU、Python 3.13/3.14；`just setup` → `just download pro-distill` → `just generate "a cat on a mat"`。
- 仓库 README 说明模型代码与 checkpoint 已开放，并列出 ComfyUI、Diffusers、vLLM-Omni 等部署入口。视频超分辨是独立仓库 [kandinsky-6-sr](https://github.com/kandinskylab/kandinsky-6-sr)。
- Wiki：[Kandinsky 6.0 Video](../../wiki/entities/paper-kandinsky-6-video.md)

## 开源边界

项目页有 “Get the code” 入口；实际运行仍需下载指定模型权重、满足 GPU/依赖要求。记录 checkpoint、采样和 offload 配置，避免把不同变体的性能混为一谈。
