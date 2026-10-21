# Norm2Tex（arXiv:2609.25398）

> 来源归档（paper）

- **标题：** Norm2Tex: Augmenting Visuo-Tactile Simulations with Texture
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.25398>
- **PDF：** <https://arxiv.org/pdf/2609.25398>
- **项目页：** <https://utn-air.github.io/norm2tex>
- **代码：** <https://github.com/utn-air/norm2tex>
- **入库日期：** 2026-10-18
- **线索来源：** [机器人会写控制程序之后，能否自己造训练数据？](../blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)（具身智能小站，2026-10-18）
- **一句话说明：** 把法线纹理转成高度信息，在触觉渲染前修改深度图，为现有模拟器增加高频表面细节。

## 开源状态

- **待发布**（步骤 2.5 核查，2026-10-18）：项目页链接的 github.com/utn-air/norm2tex 经 git ls-remote 不可读（仓库不存在或未公开），暂按待发布处理。

## 核心摘录（公众号转述，未核验原文表格）

- 现有视触觉模拟器缺高频表面纹理，影响材料分类等任务的迁移。
- 法线纹理 → 高度信息 → 修改深度图 → 触觉渲染。
- 指标：公众号称通过材料分类与控制任务检查纹理信息能否迁移到真机；未给具体数字。

## 对 wiki 的映射

- [paper-norm2tex](../../wiki/entities/paper-norm2tex.md)
- [编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-18-26-papers-technology-map.md)
