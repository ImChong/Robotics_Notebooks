# HumanTouch（Xspark AI · SparkLAB 项目页）

- **标题：** HumanTouch: A Multimodal System for Scalable Human-Hand Tactile Acquisition
- **类型：** site / project-page
- **URL：** <https://xsparkai.com/sparklab/humantouch/>
- **Lab 入口：** <https://xsparkai.com/sparklab/>（SparkLab@Xspark AI）
- **公司页：** <https://xsparkai.com/>
- **发布日期：** 2026-08-07
- **入库日期：** 2026-08-07
- **复核日期：** 2026-10-25
- **联系：** sparklab@xsparkai.com
- **项目负责人：** Chuqiao Lyu
- **核心成员：** Chenze Yu, Eric J Chen, Wenxuan Zhu
- **通讯作者：** Wenbo Ding, Tianxing Chen, Qi Xiong
- **数据集：** [ModelScope · Xspark-HumanTouch](https://www.modelscope.cn/datasets/chuqiaoLyu/Xspark-HumanTouch)；[Hugging Face · Xspark-HumanTouch](https://huggingface.co/datasets/chuqiaoLyu/Xspark-HumanTouch)（同步中）
- **代码：** 项目页未列独立 HumanTouch 采集/处理代码仓；关联策略论文 Spark-0 的模型代码已公开
- **相关论文：** [Spark-0, arXiv:2610.22342](https://arxiv.org/abs/2610.22342)（使用 HumanTouch 数据进行触觉预训练，不是 HumanTouch 采集系统的独立论文）

## 一句话摘要

Xspark AI SparkLAB 提出的 **人手可穿戴多模态触觉采集系统**：双手套压阻触觉（每手约 **360** 点）+ MANUS EMF 手姿 + 腕部 6-DoF + 头/腕多视角 RGB，强调 **可校准、可追溯、可质控** 的规模化接触数据，而非单纯堆小时数。

## 开源状态（复核于 2026-10-25）

| 项 | 状态 | 证据 |
|----|------|------|
| HumanTouch 独立采集代码 | **未列** | 项目页无 HumanTouch 专属 GitHub/代码链接；关联 Spark-0 策略代码另见其仓库 |
| GitHub org | 存在但无 HumanTouch 仓 | <https://github.com/XsparkAI>（可见 X-One-Pipeline 等，非本项目） |
| 数据集 | **已公开（首批）** | 官方页链接 ModelScope；Hugging Face 数据集仓库存在且卡片注明仍在上传。首批约 **100 h**；Spark-0 论文报告总规模 **1,000 h** |
| 数据集内容核验 | **首批切片公开，镜像同步中** | HF 页面当前约 **2.09 TB / 52,826 rows**；数据卡列出 10 个规范任务与约 100 h 首发切片 |
| 源码运行时序图 | **不适用** | 无可运行官方实现入口 |

结论：HumanTouch 首批数据集已公开；独立采集代码未列。不要将 Spark-0 模型仓误标为采集系统代码。

## 公开信息要点（复核于 2026-10-25）

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
