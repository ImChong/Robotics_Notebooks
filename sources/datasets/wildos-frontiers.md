# WildOS Frontiers Dataset（Hugging Face）

> 来源归档（dataset）

- **标题：** WildOS Frontiers Dataset
- **类型：** dataset / outdoor navigation / visual frontier annotations
- **Hugging Face：** <https://huggingface.co/datasets/leggedrobotics/wildos>
- **来源项目：** [WildOS](../../wiki/entities/paper-wildos-open-vocabulary-object-search.md)
- **底层图像：** GrandTour Dataset
- **论文：** <https://arxiv.org/abs/2602.19308>
- **许可证：** Apache-2.0（HF dataset card）
- **核查日期：** 2026-10-13
- **一句话说明：** 为户外长程导航训练视觉 frontier 预测的图像标注集，不是完整的机器人多传感器日志。

## 数据内容

数据卡描述此数据集基于 GrandTour 户外图像，标注步道尽头、树间开口或道路转弯等候选探索区域，使视觉模型能在几何深度视野之外推理。

| 子目录 | 内容 | 数量（数据卡口径） |
|------|------|------|
| annotations/ | frontier 起止像素坐标 JSON 标注 | 362 |
| RGB_frames/ | GrandTour 原始 RGB 图像及 metadata.json | 390 |
| RGB_rectified/ | 去畸变 RGB 图像 | 390 |
| SAM_boundaries/ | SAM-2 边界掩码，用于消融 | 390 |

362/390 张图像有人工标注。HF dataset viewer 显示约 1.17K rows，这是展开后的样本行数，与文件数量口径不同。页面核查时数据体积约 3.18 GB。

## 标注格式与使用提示

- 每个 annotation JSON 是 frontier 列表，含 label（目前统一为 frontier）、start: [x,y] 和 end: [x,y]。
- metadata.json 将发布图像索引映射到 GrandTour 原始图像路径。
- 单靠该数据集不能复现 WildOS 的语言目标搜索、粒子滤波定位或实地导航实验。
- 数据卡含 Python 示例，演示加载 JSON 标注、打开对应 rectified RGB 图像并绘制标注框。

## 关联资料

- [WildOS 论文归档](../papers/wildos_arxiv_2602_19308.md)
- [WildOS 项目页归档](../sites/wildos-project.md)
- [WildOS 官方仓库归档](../repos/nebula2-wildos.md)

## 一手来源

- <https://huggingface.co/datasets/leggedrobotics/wildos>
- <https://github.com/nasa-jpl/nebula2-wildos>
- <https://arxiv.org/abs/2602.19308>
