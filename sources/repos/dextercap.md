# DexterCap（PKU-MoCCA/dextercap）

> 来源归档

- **标题：** DexterCap
- **类型：** repo
- **来源：** Peking University / Tencent Robotics X
- **链接：** <https://github.com/PKU-MoCCA/dextercap>
- **项目页：** <https://pku-mocca.github.io/Dextercap-Page/>
- **论文：** <https://arxiv.org/abs/2601.05844>
- **数据：** <https://huggingface.co/datasets/pku-mocca/DexterHand>
- **许可：** 仓库未见许可证声明
- **入库日期：** 2026-07-28
- **一句话说明：** 从多视角 2D 标记、三角化到 MANO/物体拟合和 `.npz` 打包的 DexterCap 重建代码。
- **开源状态：** **部分开源**；代码和重建数据可用，原始多视角视频与中间 marker track 未发布，不能直接端到端复跑官方采集。
- **沉淀到 wiki：** [`paper-notebook-dextercap.md`](../../wiki/entities/paper-notebook-dextercap.md)

## 仓库概况（2026-07-28）

| 阶段 | 入口 |
|------|------|
| 2D 标签 | `VideoProcess/` |
| 三角化 | `MocapSystem/` |
| 手重建 | `python -m HandReconstruction.main` |
| 物体重建 | `ObjectReconstruction/` |
| 打包 | `python -m Dataset.generate_dataset ...` |
| 可视化 | `python -m Dataset.visualize --data_path ...` |

环境为 Python 3.10；MANO 模型需另行注册下载。Hugging Face 数据包含最终 `.npz` 参数，不包含原始视频。

## 对 wiki 的映射

- 项目页：[`dextercap.md`](../sites/dextercap.md)
- 论文来源：[`humanoid_pnb_dextercap.md`](../papers/humanoid_pnb_dextercap.md)
- 遥操作路线：[`depth-teleoperation.md`](../../roadmap/depth-teleoperation.md)
