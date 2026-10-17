# NavAble: A Large-Scale Dataset and Synthetic Data Generation Pipeline for Blind Navigation（arXiv:2610.19330）

> 来源归档（ingest）

- **标题：** NavAble: A Large-Scale Dataset and Synthetic Data Generation Pipeline for Blind Navigation
- **arXiv：** <https://arxiv.org/abs/2610.19330>（PDF：<https://arxiv.org/pdf/2610.19330>；HTML：<https://arxiv.org/html/2610.19330v1>）
- **数据集 / 管线：** <https://huggingface.co/datasets/NavAble/NeurIPS_2026_BLV>（归档见 [navable-project](../sites/navable-project.md)）
- **项目页：** 无独立项目页；HF 数据集页为公开入口（2026-10-17 核查）
- **作者：** Hochul Hwang、Jahir Sadik Monon、Soowan Yang、Shiven Umeshbhai Patel、Anh Nhat Hong Nguyen、Kien Nguyen、Keshav Garg、Dylan Gage、Duretti Hordofaa、Anshu Anjna、Eshed Ohn-Bar、Donghyun Kim；单位：UMass Amherst、大邱庆北科学技术院（DGIST）、波士顿大学（来自 arXiv HTML）
- **入库日期：** 2026-10-17
- **来源：** 具身智能小站 2026-10-17 公众号摘要 + arXiv HTML（前 100k 字符）
- **一句话说明：** 面向盲人 / 低视力导航的无障碍关键物体（按钮、扶手等）分割数据集，加合成数据生成管线。

## 开源状态（步骤 2.5）

- **HF 核查（2026-10-17）：** 数据集页存在，cc-by-4.0，parquet，三个子集 real_curated 36.5k、real_ours 5.58k、syn 453k 行；标签含 isaacsim；抓取内容未显示文件列表与数据卡正文，管线代码是否在文件中**未核实**。
- **论文声明：** 数据集、资产库、Isaac Sim 扩展与管线链接于上述 HF 地址；清单另引匿名仓库 anonymous.4open.science/r/NavAble-C88D（评审用）。
- **结论：** **已开源（数据集）**，管线代码声明公开但文件内容未核实。

## 摘录 1：数据与管线

- 11 类：电梯、电梯按钮、门按钮、人行横道、行人信号、APS 按钮、公交站、公交站牌、扶手、扶梯、旋转闸机（闸机无真实实例，不计入均值）。
- 真实图像约 41.5K：约 36K 来自开源数据集整理，约 5.5K 自采（RealSense D435，胸高，50 个地点）；真实测试集 1,482 张，按地点与训练分离。
- NavAble-syn：37 个环境、500+ 个 3D 资产、488 个摆放位置、25K+ 条轨迹、452K 帧、781K 实例标注；每帧含 RGB、语义分割、2D/3D 框、深度、法线。
- 管线：网页按类别爬图 → VLM（gemini-3-flash-preview）过滤图表 / CAD / 遮挡 → Grounding DINO + NMS + SAM2.1 → SAM 3D Objects 重建网格并导出 USD → Isaac Sim 扩展中遥操作录制相机轨迹并跨环境重放。

**对 wiki 的映射：** 核心原理、工程实践。

## 摘录 2：结果

- 真实测试集 mIoU（仅真实 → 最佳配置）：SAN 62.07 → 68.12；Mask2Former 68.84 → 72.67；SegFormer 61.06 → 65.96；DeepLabv3+ 与 EncNet 加入合成 / 整理数据后反而下降或持平；SAN 零样本仅 14.94。
- 摘要称合成数据最多 +3.5 mIoU，而表 3 中 SAN +6.05、SegFormer +4.90 为「最佳配置」含整理数据，口径不同，需对照原表核实。
- 两个移动辅助平台（出分布）：手杖形设备 157 张 5 类，SegFormer 11.94 → 16.17；Unitree Go2 155 张 9 类，21.56 → 23.08（部分类别仅 1 个真值实例）。

**对 wiki 的映射：** 实验与评测、局限。

## 摘录 3：局限（作者自述）

类别体系遗漏许多 BLV 相关对象；真实数据地理多样性有限；物体摆放与轨迹录制仍需人工；与辅助导航系统联合部署未验证；因算力未报告多随机种子误差棒。

## 建议 wiki 动作

- 新建 [`wiki/entities/paper-navable.md`](../../wiki/entities/paper-navable.md)。
