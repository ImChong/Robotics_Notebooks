# HumanTouch（Xspark AI · SparkLAB 项目页）

- **标题：** HumanTouch: A Multimodal System for Scalable Human-Hand Tactile Acquisition
- **类型：** site / project-page
- **URL：** <https://xsparkai.com/sparklab/humantouch/>
- **Lab 入口：** <https://xsparkai.com/sparklab/>（SparkLab@Xspark AI）
- **公司页：** <https://xsparkai.com/>
- **发布日期：** 2026-08-07
- **入库日期：** 2026-08-07
- **联系：** sparklab@xsparkai.com
- **项目负责人：** Chuqiao Lyu
- **核心成员：** Chenze Yu, Eric J Chen, Wenxuan Zhu
- **通讯作者：** Wenbo Ding, Tianxing Chen, Qi Xiong
- **数据集托管（宣称）：** [Hugging Face · XsparkAI](https://huggingface.co/XsparkAI)
- **代码：** 截至入库日项目页**未列** GitHub / 训练仓链接
- **配套论文：** 截至入库日项目页**未挂** arXiv / PDF

## 一句话摘要

Xspark AI SparkLAB 提出的 **人手可穿戴多模态触觉采集系统**：双手套压阻触觉（每手约 **360** 点）+ MANUS EMF 手姿 + 腕部 6-DoF + 头/腕多视角 RGB，强调 **可校准、可追溯、可质控** 的规模化接触数据，而非单纯堆小时数。

## 开源状态（步骤 2.5 · 2026-08-07 核查）

| 项 | 状态 | 证据 |
|----|------|------|
| 项目页 Code 链 | **无** | 页内仅 HF org 与 mailto；无 GitHub Code 按钮 |
| GitHub org | 存在但无 HumanTouch 仓 | <https://github.com/XsparkAI>（可见 X-One-Pipeline 等，非本项目） |
| 数据集 | **宣称将开源 / 待发布** | 文称渐进发布于 HF；首批约 **100 h** 目标 **2026-08-15**，随后扩至 **1,000 h** |
| HF org 公开资产 | **空** | 截至入库日 `XsparkAI` 公开 models/datasets = 0 |
| 源码运行时序图 | **不适用** | 无可运行官方实现入口 |

结论写入 wiki：按「宣称将开源数据、代码未列」处理；勿写「已开源可复现」。

## 公开信息要点（截至入库日）

- **传感取舍：** 选用柔性压阻手套以覆盖指面+手掌；承认非线性、迟滞、漂移与姿态伪迹，故配套姿态/历史感知标定与生命周期质控。
- **手姿：** MANUS **EMF** 跟踪（相对 IMU 无积分漂移；金属干扰需控场）；触觉自接触用于校准指段长度/关节偏置。
- **同步：** 离线对齐到统一 **60 Hz** 参考时间线；位置线性插值、旋转 SLERP、图像最近邻；缺模态/大间隙直接拒收。
- **表征：** 税点聚类为解剖 **tactile patch**；输出接触置信度、相对强度、响应中心与不确定性——**不是**物理力/载荷标定。
- **对比表（项目页 Table 1）：** 相对 OpenTouch / World In Your Hands / FreeTacMan / Touch in the Wild，宣称：人手平台、360 点/手、头+腕多视角、**1000+ h / 100+ tasks**（全库目标）、60 Hz、有力标定流程、多站点部署。
- **初版发布切片：** 约 **100 h**、十个规范任务、约 **13,469** episode；另预告 AIGC 版。
- **质量指标 DcSNR：** 十任务点估计约 **3.61–7.19 dB**；32 操作者 1 h 轨迹中位数约 **7.07–8.00 dB**（时间标准差 0.19 dB）。

## 为何值得保留

- 把「规模化触觉」问题从堆小时数改写为 **可解释接触 + 校准/质控协议**，直接服务接触丰富模仿学习与视触觉预训练选型。
- 与站内 [OSMO 触觉手套](../../wiki/entities/paper-notebook-osmo-open-source-tactile-glove-for-human-to-robo.md)、[灵巧采数指南](../../wiki/queries/dexterous-data-collection-guide.md) 形成 **人侧全掌压阻 vs 人机共用磁触觉** 对照。

## 关联资料

- wiki：[`wiki/entities/humantouch.md`](../../wiki/entities/humantouch.md)
- 机构：Xspark AI · SparkLAB（见 `schema/institutions.json` → `xspark-ai`）
