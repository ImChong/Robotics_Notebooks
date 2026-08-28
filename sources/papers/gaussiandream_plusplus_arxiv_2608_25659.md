# GaussianDream++: Efficient 3D Gaussian World Modeling for Robotic Manipulation

> 来源归档（ingest）

- **标题：** GaussianDream++: Efficient 3D Gaussian World Modeling for Robotic Manipulation
- **短名：** GaussianDream++
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2608.25659>
- **PDF：** <https://arxiv.org/pdf/2608.25659>
- **项目页：** <https://tuojingai.github.io/GaussianDream-Series-project-page/>
- **代码：** <https://github.com/TuojingAI/GaussianDream>
- **入库日期：** 2026-08-28
- **索引来源：** [具身智能小站 9 篇盘点](../blogs/wechat_embodied_station_wam_vla_cross_embodiment_9_papers_2026-08-28.md)（<https://mp.weixin.qq.com/s/FNhRO3KOm8k8CkJEqystQQ>）
- **一句话说明：** 把当前世界与未来预测压缩进 20 个世界令牌，推理时无需在线高斯解码。
- **消歧：** 与 Awesome 索引级 [GaussianDream](../../wiki/entities/paper-sa-2605-20752-gaussiandream-a-feed-forward-3d-gaussian-world-m.md)（[arXiv:2605.20752](https://arxiv.org/abs/2605.20752)）是**不同论文**，本页只覆盖 ++。

## 开源状态（步骤 2.5）

- **部分开源**：[`TuojingAI/GaussianDream`](https://github.com/TuojingAI/GaussianDream) 已有 v1 训练/评测实现（Apache-2.0）；README 标题与 arXiv 链接仍指向 **2605.20752**，徽章写 Release Coming Soon。++ 论文把同一仓列为 Code URL，但截至入库日 README **未标明** World State / Prediction Tokens 的独立入口或 ++ 权重。

## 核心摘录（面向 wiki 编译）

### 摘录 1：20 个世界令牌 + 训练期高斯头

- 在 VLA 主干插入 World State Tokens 与 World Prediction Tokens；训练期 World Representation Head 解码为共享高斯基元上的当前世界与未来预测。
- 静态—动态分解聚焦交互区域；推理时移除表征头、渲染器、辅助目标及 VGGT/TGE 路径。

**对 wiki 的映射：** [paper-gaussiandream-plusplus](../../wiki/entities/paper-gaussiandream-plusplus.md)、[生成式世界模型](../../wiki/methods/generative-world-models.md)

### 摘录 2：评测

- LIBERO **98.6%**、LIBERO-Plus **87.8%**；相对 GaussianDream 总体 +0.8 pp，Camera / Layout 偏移 +2.8 / +1.6 pp。
- 真机平均成功率由复现 π0.5 的 **29.2%** 提升至 **52.5%**。

**对 wiki 的映射：** [libero-benchmark](../../wiki/entities/libero-benchmark.md)

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-gaussiandream-plusplus.md`](../../wiki/entities/paper-gaussiandream-plusplus.md)

## 当前提炼状态

- [x] 方法要点与开源核查
- [x] wiki 实体与技术地图回链
