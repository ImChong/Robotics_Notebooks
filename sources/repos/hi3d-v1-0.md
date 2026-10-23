# Hi3D v1.0 官方推理仓库

- **标题：** Hi3D v1.0
- **类型：** repo / 3D 几何细化模型
- **来源：** Hi3D 官方组织
- **链接：** <https://github.com/hi3d-official/Hi3Dv1.0>
- **项目页：** <https://www.hi3d.ai/hi3d-1-0-open-source>
- **模型权重：** <https://huggingface.co/hi3d-official/Hi3Dv1.0>
- **在线体验：** <https://huggingface.co/spaces/ilcve21/Sparc3D>
- **入库日期：** 2026-10-23
- **一句话说明：** 基于 Sparc3D 的高分辨率 3D 几何细化模型；以同一物体的参考图和粗网格为条件，逐级生成更细的网格几何。

## 开源状态核查（2026-10-23）

| 项 | 结论 |
|----|------|
| **代码** | **已开源** — 官方仓库提供推理代码、示例与文档；不包含训练代码和论文评测工具。 |
| **权重** | **已公开** — 模型权重在官方 Hugging Face 仓库，README 列出模型配置和 safetensors 文件。 |
| **训练数据** | README 报告训练使用约 200 万精选 3D 资产；该仓库未列出可下载训练集。 |
| **在线体验** | 官方 README 将 `ilcve21/Sparc3D` 列为 Live Demo。Space 名称仍为 Sparc3D，部署内容和可用性可能变化，使用前应核对页面当前版本。 |
| **许可** | Hi3D Open License Agreement Version 1.0。协议允许商业使用，但达到规定的月活或月收入门槛时需在 60 日内通知并申请单独书面许可；商业集成还须按协议展示 “Powered by Hi3D 1.0”。具体范围、门槛计算和第三方组件条款以仓库 LICENSE 与 THIRD_PARTY_LICENSES.md 为准。 |

## 方法与运行入口

官方 README 将 Hi3D v1.0 描述为 Sparc3D 表示与细化架构上的高分辨率几何细化：从粗网格确定的稀疏坐标出发，图像特征为逐级稀疏潜变量采样提供条件，再由自剪枝稀疏 VAE 解码并提取网格。可按 512³、1024³ 或 1536³ 的稀疏提取网格档位运行，数值并不代表稠密潜变量张量或固定三角面数。

- **Python API：** `Sparc3DPipeline.from_pretrained("hi3d-official/Hi3Dv1.0")`
- **CLI：** `demo/refine_mesh.py` 可对已有粗网格做细化；`demo/image_to_mesh.py` 是可选的端到端示例。
- **可选前级：** 端到端示例使用 Hunyuan3D-2 mini 产生粗网格；它是独立第三方依赖，不属于 Hi3D 核心细化器。
- **实测环境：** Linux、Python 3.10、NVIDIA GPU 计算能力 8.0+；README 给出的验证栈使用 CUDA 12.4、PyTorch 2.5.0 与 96 GB H20。1536³ 档仅报告在该卡上验证。

## 对知识库的映射

- 升格页面：[wiki/entities/hi3d-v1-0.md](../../wiki/entities/hi3d-v1-0.md)
- 关联资产生成：[wiki/entities/3dgenstudio.md](../../wiki/entities/3dgenstudio.md)、[wiki/entities/meshroom.md](../../wiki/entities/meshroom.md)、[wiki/concepts/sim2real.md](../../wiki/concepts/sim2real.md)
- 项目页归档：[sources/sites/hi3d-v1-0-open-source.md](../sites/hi3d-v1-0-open-source.md)

## 参考链接

- GitHub：<https://github.com/hi3d-official/Hi3Dv1.0>
- 官方开源介绍：<https://www.hi3d.ai/hi3d-1-0-open-source>
- Hugging Face 权重：<https://huggingface.co/hi3d-official/Hi3Dv1.0>
- Hugging Face Space：<https://huggingface.co/spaces/ilcve21/Sparc3D>
- Sparc3D 论文：<https://arxiv.org/abs/2505.14521>
