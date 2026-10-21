# VisionHOPE 官方代码仓库

> 来源归档（ingest）

- **标题：** VisionHOPE: Visual Backbones as Self-Modifying Learning Systems
- **类型：** repo
- **仓库：** <https://github.com/PSRben/VisionHOPE>
- **许可证：** MIT
- **主论文：** <https://arxiv.org/abs/2609.33325>
- **模型权重：** <https://huggingface.co/PSRben/VisionHOPE>
- **核查日期：** 2026-10-18
- **一句话说明：** 官方 PyTorch 实现，含 ImageNet 分类、COCO 检测/实例分割、ADE20K 语义分割训练与评估脚本，以及 SRNL 和 VisionHOPE operator/block 模块。
- **对 wiki 的映射：** [VisionHOPE 论文实体页](../../wiki/entities/paper-visionhope.md)

## 开源范围核查

README 提供模型定义、训练/测试脚本、环境安装步骤和 standalone 模块用法；预训练权重通过 GitHub Releases 与 Hugging Face 发布。仓库 README 要求 Linux、NVIDIA GPU、CUDA toolkit（含 nvcc）与 C++ 编译器；Python 3.10 为推荐版本，分类另支持 Python 3.11。论文页和仓库 README 均指向同一官方实现。
