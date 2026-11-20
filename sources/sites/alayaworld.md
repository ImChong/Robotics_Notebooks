# AlayaWorld — 官方主页与开源核验

- **名称：** AlayaWorld
- **类型：** site / project
- **项目页：** <https://alaya-lab.github.io/AlayaWorld/>
- **代码：** <https://github.com/AlayaLab/AlayaWorld>
- **技术原文：** <https://arxiv.org/abs/2607.18367>
- **机构：** Alaya Lab（AlayaWorld Team）
- **核对日期：** 2026-11-20
- **核对范围：** 官方仓库 README 链接、发布内容及对应介绍
- **通读：** 已完成

## 技术定位

在双向视频基础模型预训练后，转入因果视频块生成：条件前缀由固定 sink、压缩历史特征、按当前相机从空间记忆重投影的条件和最近帧组成；用户可逐块切换提示。训练通过模型自身 rollout 的退化样本缓解积累漂移，后训练用 DMD + Self-Forcing++ + 一致性蒸馏把约 30 步缩到 4 步。仓库 v1.1 额外提供 DA3 与 ViGeo 两套空间记忆实现，需对齐所用 checkpoint。

## 开放状态和复现提示

- 推理、训练、checkpoint 已发布，外部依赖/门控模型和许可证需要单独核对。
- 部署涉及大权重、DA3/ViGeo 外部依赖与受门控的 Gemma 文本编码器，不能把“开放代码”解读为所有依赖即取即用。论文多版本 checkpoint 功能范围不同，不能混用；生成一致性不能证明符合真实碰撞动力学。
- 入口与源码分层见 [归档](../repos/alayaworld.md)。

## 映射

- [项目/论文详情](../../wiki/entities/paper-alayaworld.md)
