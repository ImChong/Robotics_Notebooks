# DreamX：视频扩散模型作为通用潜动作学习器

> 来源归档（paper）

- **标题：** DreamX: Video Diffusion Model is a Universal Latent Action Learner
- **类型：** paper
- **作者：** Jisoo Kim、Chaehyun Kim、Junsu Kim、Jiho Park、Seongchan Kim、Eunbeen Hong、Soowon Son、Fu-En Yang、Seungryong Kim
- **机构：** 韩国科学技术院（KAIST）、英伟达（NVIDIA）
- **arXiv：** <https://arxiv.org/abs/2611.05377v1>
- **PDF：** <https://arxiv.org/pdf/2611.05377v1>
- **HTML：** <https://arxiv.org/html/2611.05377v1>
- **项目页：** <https://cvlab-kaist.github.io/DreamX/>
- **项目页归档：** [dreamx.md](../sites/dreamx.md)
- **版本：** v1，2026-11-03
- **入库日期：** 2026-11-05
- **核对日期：** 2026-11-24
- **核对范围：** arXiv HTML v1 正文 §1–5、伦理与复现声明、附录 A–D 的实现、实验协议、补充结果和分析；官方项目页开源入口。
- **通读：** 已完成
- **线索来源：** [具身智能小站 2026-11-05 盘点](../blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)
- **一句话说明：** 复用视频扩散先验提取共享潜动作，实现跨本体视频及控制迁移。

## 开源状态

- **待发布**：2026-11-24 项目页仍显示 Code soon；未发现官方 DreamX 训练或推理代码、权重入口。附录 A.1 所列训练来源为公开数据集，不等于本模型已开放。

## 核心摘录（依据原文归纳）

- §3.1–3.2、图 2：Cosmos-Predict2.5 2B 读出每潜帧四个 query，首帧条件的未来重构与可选动作监督联合训练；1200→4 指 token 数。
- §3.3–3.4、图 3：源视频的潜动作可结合目标首帧或目标机器人状态；下游策略预测 32 维压缩码与动作而非未来视频。
- §4 表 1–3：LARYBench MSE 0.17、分类 69.67%；H&R FVD 57.44；关抽屉平均子目标成功率 96.7%，单次推理 1474 ms。
- 附录 A–B：2190 小时、六个数据集；双臂 YAM 14 维绝对动作，开环人机迁移与观测条件策略分开；关抽屉每配置 10 次测试。
- 附录 C–D：动作监督消融、CALVIN 条件消融、扰动和可视化支持共享运动结构，未证明任意本体或任意任务均可安全执行。

## 对 wiki 的映射

- [paper-dreamx](../../wiki/entities/paper-dreamx.md)
