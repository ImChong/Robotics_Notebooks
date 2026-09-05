# Cosmos-Transfer2.5 官方文档

> 来源归档

- **标题：** Cosmos-Transfer2.5
- **类型：** site（NVIDIA Docs）
- **来源：** NVIDIA
- **链接：** https://docs.nvidia.com/cosmos/latest/transfer2.5/index.html
- **文档版本：** 2.2.0（抓取日）
- **入库日期：** 2026-09-05
- **一句话说明：** Transfer2.5 产品文档首页：多 ControlNet 吃 RGB / depth / segmentation 等视频模态；JSON `controlnet_specs` 配权重；两条增广叙事是 **Simulations to Photorealism** 与 **Scale World State Diversity**。
- **沉淀到 wiki：** 是 → [`wiki/entities/cosmos-transfer.md`](../../wiki/entities/cosmos-transfer.md)

## 开源边界（步骤 2.5）

文档描述的可运行实现在 [nvidia-cosmos/cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5)（~733★）。→ **已开源**；细节见 [`sources/repos/nvidia_cosmos_transfer25.md`](../repos/nvidia_cosmos_transfer25.md)。

## 页面要点（2026-09-05）

- 支持单视频推理、**控制图自动生成**、多 GPU。
- **Simulations to Photorealism：** 降低对 3D 仿真「像素级保真」的依赖。示例：双臂在实验室操作蓝布 → 计算控制图 → 照片级输出；长文本 prompt 描述场景与动作。
- **Scale World State Diversity：** 用传感器 RGB 或 GT 增广。示例：城市场景 dashcam → 控制图 → 天气 / 外观多样输出。

可运行命令与 JSON 规格以仓库 `docs/inference.md` 为准，不以此营销页为唯一入口。

## 对 wiki 的映射

- Transfer 族：[`wiki/entities/cosmos-transfer.md`](../../wiki/entities/cosmos-transfer.md)
- 2.5 论文：[`wiki/entities/paper-sa-2511-00062-world-simulation-with-video-foundation-models-fo.md`](../../wiki/entities/paper-sa-2511-00062-world-simulation-with-video-foundation-models-fo.md)
