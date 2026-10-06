# UniWAM 官方代码仓库

> 来源归档（repo；核对官方仓库 README 与项目页；2026-10-06）

- **类型：** research-code / unified world-action model
- **仓库：** <https://github.com/UniWAM/UniWAM>
- **项目页：** <https://uniwam.github.io/>
- **论文：** <https://arxiv.org/abs/2610.02054>
- **License：** Apache-2.0
- **权重：** [ModelScope UniWAM collection](https://www.modelscope.cn/collections/Kosmos524/UniWAM)
- **依赖骨干：** Qwen3-VL-2B-Instruct；Wan2.2-TI2V-5B；官方 README 将整套 UniWAM 描述为 8B 参数。
- **代码目录：** `models/`、`train/`、`utils/`；RoboTwin 2.0 post-training/inference 位于 `inference/robotwin/uniwam/`；LIBERO 训练和 LIBERO / LIBERO-Plus 评测示例也已提供。
- **硬件提示：** README 报告 RoboTwin 2.0 后训练约 10 小时、8 张 NVIDIA H100；安装建议 Python 3.10、CUDA-capable PyTorch、FlashAttention。
- **当前范围：** README 明确说明 Bridge、DROID、Fractal 以及 real-world inference 不在当前 release 范围；论文中的双臂实机结果不等于仓库已有可直接复现的真实机器人部署流程。

## 关键复现边界

- ModelScope 页面链接的是 UniWAM 检查点集合；训练配置还需要 Qwen3-VL 与 Wan2.2 视频 backbone 等预训练资产。
- 仓库提供的是模型实现、训练/仿真评测资源，不代表论文表中所有原始数据集都被仓库重新分发。
- 对照数字需按论文中的 benchmark、clean/randomized split 和评测 protocol 阅读；不要把 LIBERO、LIBERO-Plus、RoboTwin 与真机任务视为同一设置。

## 沉淀到 Wiki

- [UniWAM 独立详情节点](../../wiki/entities/paper-uniwam-unified-world-action-model.md)
- [论文来源归档](../papers/uniwam_arxiv_2610_02054.md)
