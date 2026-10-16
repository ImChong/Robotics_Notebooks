# GATOR — NVIDIA Learning and Perception Research 项目页

> 来源归档（site）

- **标题：** GATOR: Generative and Agentic 3D Object Reconstruction From Casual Images
- **类型：** site / 论文项目页
- **作者：** Qirui Wu、Stan Birchfield、Hesam Rabeti、Angel X. Chang、Bowen Wen
- **机构：** NVIDIA Learning and Perception Research；西蒙菲莎大学（Simon Fraser University）
- **链接：** <https://research.nvidia.com/labs/lpr/gator/>
- **论文：** <https://arxiv.org/abs/2610.11215>
- **入库日期：** 2026-10-16
- **一句话说明：** GATOR 将随手拍图像中的稀疏观察转为有纹理且场景对齐的 3D 物体，并用图像证据驱动的 agent 在 Blender 中修复资产。
- **代码：** **待发布**（页面标注 `Code (Coming Soon)`；截至 2026-10-16 无仓库链接）
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-gator.md`](../../wiki/entities/paper-gator.md)

## 页面要点（2026-10-16 核查）

- **输入与输出：** 一张或多张 RGB 图像及目标 mask、pointmap 等空间线索；输出完整物体资产与场景相对位姿。
- **生成：** patch 对齐的 RGB / mask / pointmap 特征先混合再跨视角推理；类别文本负责结构线索，图像特征与对象描述补充几何和外观。
- **精修：** 图像接地的 edit-render-review 闭环在 Blender 中做局部网格与纹理修改；对照相同相机和光照下的渲染，选择保留或撤销。
- **评测与演示：** 页面展示 Toys4K 有限时间比较、ScanNet++ 与 HANDAL 场景重建、Isaac Sim 房间沉降，以及人形机器人与重建场景交互。
- **复现边界：** 页面当前未提供代码、数据或权重下载链接；演示不等同于可复现发行包。

## 对 wiki 的映射

- [GATOR 详情页](../../wiki/entities/paper-gator.md)
- [论文摘录](../papers/gator_arxiv_2610_11215.md)
