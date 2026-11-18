# Code World Model 2.0 / Invoker-Render 官方 GitHub 仓库归档

> 原始项目资料；核查日期：2026-11-18

- **项目：** Code World Model 2.0: Towards Infinite Controllable Video Generation
- **组织 / 机构：** Inclusion AI AGI Research Center（蚂蚁集团）、西湖大学、浙江大学
- **作者：** Ruoyu Wang, Hao Zhong, Yanming Yang, Jiahao Tian, Lin Liu 等（官方 README）
- **类型：** repo（开源推理运行时与视频流服务）
- **代码：** https://github.com/inclusionAI/CodeWorldModel
- **中文 README：** https://github.com/inclusionAI/CodeWorldModel/blob/master/README.zh-CN.md
- **部署文档：** https://github.com/inclusionAI/CodeWorldModel/blob/master/docs/INFERENCE.md
- **CLI 源码：** https://github.com/inclusionAI/CodeWorldModel/blob/master/invoker_render.py
- **推理客户端：** https://github.com/inclusionAI/CodeWorldModel/blob/master/src/sglang/projects/minimax_h3/tools/infer_omni_h3.py
- **服务入口：** https://github.com/inclusionAI/CodeWorldModel/blob/master/src/sglang/projects/minimax_h3/tools/serve_omni_h3.py
- **项目页：** https://www.inclusion-ai.org/CodeWorldModel/ （[归档](../sites/inclusionai-codeworldmodel.md)）
- **模型权重：** https://huggingface.co/inclusionAI/invoker-render-1.0 （[归档](../models/invoker-render-1-0.md)）
- **技术报告：** https://github.com/inclusionAI/CodeWorldModel/blob/master/docs/CodeWorldModel2.pdf
- **核对范围：** 阅读官方 README.zh-CN、README.md 的定位、技术亮点、发布清单、快速入门、部署要求、许可说明；阅读全文档 docs/INFERENCE.md 的环境、推理协议、bundle、部署 profile、文件推理、浏览器调用与验证章节；核对 CLI 主入口 invoker_render.py、infer_omni_h3.py、serve_omni_h3.py 的调用路径。并未逐行审计 vendor 目录或 CUDA kernel。报告 PDF 仅确认其仓库存在，本次未通读；arXiv 仍标注待发布。
- **通读：** 已完成
- **注意：** “已完成”仅指本条上列明确列举的官方 README、部署文档和关键入口，**不包括**尚未读取的技术报告 PDF。
- **Wiki 映射：** [独立 Code World Model 2.0 项目节点](../../wiki/entities/code-world-model-2-invoker-render.md)

## 一手源码机制

| 范围 | 已核对边界 |
|---|---|
| 命令分发 | `invoker_render.py` 的 COMMANDS 映射 `prepare/serve/infer/web-adapter/web`，`check-env` 单独执行检测脚本；通过 `launch.py` 调用 projects.minimax_h3.tools 对应模块 |
| 推理客户端 | `infer_omni_h3.py` 参数校验后通过 `asyncio.run(infer_files(...))` 进入 `omni_h3_client`，输出帧数与 status |
| 会话服务 | `serve_omni_h3.py` 根据 bundle 元数据选择实时会话、codec overlap 与 CPU delivery，构建 H3 stage 并 `launch_server(host='127.0.0.1',port=...)` |
| 模型契约 | 部署文档定义 `cwm_causal/v1`、4 个 Euler step、9 latent 首块 / 5 latent 后续块、sink 1 / recent 12、47 latent RoPE 参考 |
| 文件输出 | 首帧 1344×768、Proxy 24 FPS、帧数 `17*N+5` 且最少 39；输出 `video.mp4` 与 `inference.json` |
| 部署边界 | production 默认单机 8 GPU，模型文件需额外准备匹配的 frozen artifact、编码器、因果解码器与 silence tensors，不附带可直接运行的完整 bundle |

## 开放状态与事实边界

- **已公开：** 源代码（仓库自有 Apache-2.0）、README/部署文档、Hugging Face DiT 权重。
- **尚未公开：** 训练方法与数据、arXiv 版技术报告（仓库包含 PDF，但本归档未通读）。
- **许可证：** 仓库自身 Apache-2.0；第三方包和单独资产以各自条款为准，不能代替 HF 模型卡的复合许可。
- **未执行：** 依赖安装、bundle 构建、8 GPU 服务启动、推理和帧率实测。服务/时序图是源码静态读取结论。
