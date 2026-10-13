# UniWAM 官方代码仓库

> 来源归档（repo；核对官方仓库 README、RoboTwin/LIBERO 指南及项目页；更新核查：2026-10-13）

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


## 本次核查补充

- 用户给出的 `UniWAM/UniWAMModelScope` 地址当前返回 404；arXiv 页面与 UniWAM 项目页的 Code 按钮指向实际官方代码 [`UniWAM/UniWAM`](https://github.com/UniWAM/UniWAM)。仓库 README、Apache-2.0 LICENSE、`models/`、`train/` 与推理目录相互吻合。
- 代码已包含 RoboTwin 2.0 数据转换、训练和 `inference/robotwin/uniwam/` 策略部署；另有 LIBERO / LIBERO-Plus 训练评估指南。模型权重需另外取得，仓库不捆绑基准场景和训练数据。
- README 把 ModelScope / Hugging Face collection 中的模型列为 `UniWAM-base` 与 `UniWAM-robotwin-clean`；ModelScope 页面本次可确认集合入口，但未能独立核验权重文件列表、版本、许可或下载状态。
- 可运行范围有边界：README 明确排除 Bridge、DROID、Fractal 和 real-world inference。8×H100、约 10 小时是 RoboTwin 后训练口径，不是完整预训练耗时。