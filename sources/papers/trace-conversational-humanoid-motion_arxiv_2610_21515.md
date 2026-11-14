# TRACE: Typed Reference-Aware Conversational Editing for Humanoid Motion Diffusion

> 来源归档（ingest），来自 2026-11-13「具身智能小站」18 篇论文阅读清单；以原始 arXiv 及官方项目和仓库为核查依据。

## 元数据

- **标题：** TRACE: Typed Reference-Aware Conversational Editing for Humanoid Motion Diffusion
- **中文：** TRACE：具有参考保持约束的对话式人形动作编辑
- **arXiv：** 2610.21515
- **版本：** 2026-11-14 检查 arXiv HTML；具体修订号待逐项补齐
- **入库日期：** 2026-11-14
- **核对范围：** 已直接阅读 arXiv HTML 摘要、方法与部分实验/局限章节；尚未逐行检查全文及全部附录，后续继续核对
- **通读：** 未完成
- **论文：** https://arxiv.org/abs/2610.21515
- **HTML：** https://arxiv.org/html/2610.21515
- **PDF：** https://arxiv.org/pdf/2610.21515
- **仓库：** https://github.com/Multimodal-Perception-Action/TRACE

## 论文机制

从文本和原动作生成稀疏运动规划条件；扩散模型在重力对齐与接触一致性约束下生成动作；针对编辑请求自动构建 body-time mask，以源动作为参考保留未指定的区域与时间段。

## 实验与证据边界

论文包含动作生成、连续多轮编辑和用户研究；真机展示属于定性验证，主要实验对运动学轨迹进行评测，不能直接当作真机闭环控制成功率。

## 复现与开放状态

官方仓库公开 Unitree G1 generation inference、模型与资源，并列有训练文档；本次尚未核查具体源码调用。

## 局限

轨迹生成质量不等价于实机动力学可执行性；真实跟踪仍需下游控制器和机器人约束。

## 对 wiki 的映射

- [TRACE：具有参考保持约束的对话式人形动作编辑](../../wiki/entities/paper-trace-conversational-humanoid-motion.md)

## 参考来源

- [原论文 HTML](https://arxiv.org/html/2610.21515)
- [转载阅读线索（仅作索引）](https://mp.weixin.qq.com/s/gatO-d6IxsemBSk62EvS1w)
