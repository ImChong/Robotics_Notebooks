# Scalabot（缩放机器人）官网

> 来源归档（ingest · 公司技术子品牌门户）

- **URL：** <https://scalabot.noetixrobotics.com/>
- **标题：** Scalabot · 缩放机器人 | 面向真实世界的通用具身智能与世界模型
- **机构：** 北京松延动力科技集团股份有限公司（Noetix Robotics）· Scalabot 品牌
- **类型：** site（公司技术品牌门户）
- **入库日期：** 2026-10-22
- **核对范围：** 已抓取并通读官方技术子品牌站点 SPA 打包资源 `https://scalabot.noetixrobotics.com/assets/index-wQMifLdw.js` 内的完整中英文官方技术博客正文（对应路由 `https://scalabot.noetixrobotics.com/blog/wm` 与 `/en/blog/wm`，`# HERON-World Model：面向具身交互的动作条件世界模型`）；同时核实截至 2026-11-21 官方未公开 HERON-World Model 模型权重或训练/推理代码仓库（`URDream-project/URDream` 仅对应 HERON-Spatial Intelligence）。
- **通读：** 已完成
- **一句话说明：** 松延动力于 2026-09-08 发布的通用具身智能技术品牌，专注世界模型与 HERON-CRA 研究；首个世界模型为 HERON-World Model。

## 归属判断（公司 vs 实验室）

- 页面 `<meta name="author">` 与 JSON-LD `parentOrganization` 均写 **北京松延动力科技集团股份有限公司**，域名为 `noetixrobotics.com` 子域；Scalabot 自身 JSON-LD 写 `Organization`、`foundingDate: 2024`（品牌 / 团队口径，非独立法人）。
- 媒体亦称其为松延动力的 **子品牌 / 技术品牌**（[Gasgoo](https://autonews.gasgoo.com/articles/news/noetix-robotics-launches-embodied-intelligence-sub-brand-scalabot-2097870821795381249)）。
- 结论：属 **公司** 路线（并入松延动力），不是独立研究实验室。

## 官方技术博客一手摘录（`scalabot.noetixrobotics.com/blog/wm`）

- **定位与目标**：Scalabot 于 2026-09-08 发布，HERON 为统一架构，**HERON-World Model** 面向具身交互中的长程、动作条件未来视频与物理规律预测（支持单/双臂、灵巧手及多智能体交互）。
- **三层数据金字塔与伪动作提取**：融合真机遥操作数据、真实重建仿真数据与人类第一视角（egocentric）视频，并按部分/全量/无动作三类监督层级分配权重；伪动作管线从人类视频提取头部相机位姿、双手腕 6DoF 与 10 指尖 3D 坐标，统一转换为锚定首帧局部坐标系下的相对增量序列 `[a_geo || chi]` 并经统一 Action Encoder 映射为 Action Tokens。
- **模型架构（VLM + 噪声路由双专家 MoE DiT）**：以冻结的预训练 **VLM** 作为物理与语义条件编码器，通过 **单向 Mixture-of-Transformers (MoT)** 联合注意力接入扩散生成骨干（VLM 侧梯度切断且 KV cache 可在多组候选动作评估中复用）；去噪骨干沿扩散噪声时间步信噪比阈值路由至高噪声专家（`lambda(tau) <= gamma`，管全局运动与交互拓扑）与低噪声专家（`lambda(tau) > gamma`，管几何细节与表面一致性），每步仅激活单个专家；全模块采用 3D 多模态 RoPE 对齐视觉、动作与语言 token。
- **评测与四类下游应用**：官方在 WorldArena 1.0 Track 1（按页面榜单说明为离线自测提交）报告综合 `EWMScore_P = 75.80`（并公布 EgoDex、AgiBot、RoboTwin 上的 13 项分量指标）；文章给出四类应用路径——闭环长程 Rollout、多候选动作评分（复用 VLM KV cache）、基于 MPC/滑动视界重规划的真机执行、反事实轨迹合成与 RL 策略评估。
- **开源现状**：截至本次核对，官方仅公开官网与技术博客图文演示，未发布 HERON-World Model 代码仓库、论文 arXiv 或模型权重；同源项目 `URDream-project/URDream` 面向 HERON-Spatial Intelligence 而非 HERON-World Model。
- **站点 `build-time`**：2026-09-23。

## 媒体报道补充（核对基准）

- 2026-09-08 发布 Scalabot；HERON 为技术架构体系，HERON-World Model 为其下首个世界模型（[Gasgoo](https://autonews.gasgoo.com/articles/news/noetix-robotics-launches-embodied-intelligence-sub-brand-scalabot-2097870821795381249)，文章 2026-09-10）。
- [量子位 2026-09-09](https://www.qbitai.com/2026/09/486291.html) 引述之 WorldArena 1.0 Track 1 分项（物理交互 95.60 / 空间视角 99.20 / 运动动力学 92.61 / 轨迹精度 56.66 / 综合 `EWMScore_P = 75.80`）与官网博客及榜单记录数值吻合，属团队自测离线口径，尚缺第三方开源复现权重验证。
- HERON-CRA 在官网门户关键词提及，尚无独立公开技术白皮书。

## 对 wiki 的映射

- [`wiki/entities/scalabot-heron-world-model.md`](../../wiki/entities/scalabot-heron-world-model.md)
