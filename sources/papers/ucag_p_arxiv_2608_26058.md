# UCAG-P（arXiv:2608.26058）

> 来源归档（ingest）

- **标题：** One Policy, Many Embodiments: Unified Camera-Centric Action Geometry Pre-training for Heterogeneous Embodied Manipulation（**UCAG-P**）
- **类型：** paper / vla / cross-embodiment / manipulation
- **arXiv：** <https://arxiv.org/abs/2608.26058>
- **PDF：** <https://arxiv.org/pdf/2608.26058>
- **HTML：** <https://arxiv.org/html/2608.26058>
- **项目页：** <https://public-bots.github.io/UCAG-P>
- **代码仓：** <https://github.com/Public-BOTs/UCAG-P>（截至 2026-08-28 仅为项目页/配图；README 写 *The training, inference, and evaluation code will be released soon.*）
- **机构：** 小米具身智能团队（Xiaomi Embodied Intelligence Team）；澳门大学（University of Macau）
- **作者：** Shaoqing Xu, Fang Li, Guozhi Zhan, Zhixiang Duan, Yuhan Wang, Yuechen Luo, Shengyin Jiang, Hanbing Li, Zhiying Du, Hangjun Ye, Zhi-xin Yang, Longlong Wang, Longmei Jiang, Weixiang Liang, Ying Gong, Yong Pan, Ziping Zhao, Zhiyuan Chen, Yangwei You, Kun Ma, Qinyuan Liu
- **核心贡献者：** Shaoqing Xu\*†, Fang Li\*, Guozhi Zhan\*, Zhixiang Duan\*†, Yuhan Wang, Yuechen Luo, Shengyin Jiang, Hanbing Li, Zhiying Du, Hangjun Ye, Zhi-xin Yang ✉（\* 同等贡献；† Project Leads）
- **入库日期：** 2026-08-28
- **索引来源：** 独立 ingest；亦见于 [具身智能小站 9 篇盘点](../blogs/wechat_embodied_station_wam_vla_cross_embodiment_9_papers_2026-08-28.md)
- **一句话说明：** 把异构机器人与人手演示对齐到 **相机系腕/抓取锚点运动**，共享 VLA 学几何、几何条件翻译器再出各本体可执行命令；单 checkpoint 跨 LIBERO / RoboTwin / RoboCasa GR-1，无需 per-benchmark 微调。

## 开源状态（步骤 2.5，2026-08-28）

| 资源 | 状态 |
|------|------|
| 项目页 | **已发布** — 演示视频、系统总览、仿真/真机数字、BibTeX |
| GitHub | **宣称将开源 / 待发布** — [Public-BOTs/UCAG-P](https://github.com/Public-BOTs/UCAG-P) 现为项目页与 `assets/` 配图仓；无训练/推理入口、无权重、无许可证 |
| 预训练权重 / 数据集打包 | **未列下载** |
| 论文承诺 | 摘要写 *Project Page & Code*；README *Code — Release Soon* |

勿把 GitHub Pages 仓当成可复现训练栈。开放后应补 `sources/repos/` 运行入口并在实体页补源码时序图。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / §1）

- **核心贡献：** 通才 VLA 的瓶颈不只是数据规模，而是 **形态 / 相机 / 低层动作空间异构**。现有路线多用 per-dataset 动作头、embodiment prompt、相机系 EEF 对齐，或先做人→机视频合成/重定向。UCAG-P 主张：**最稳定的共享结构不在控制器，而在相机可观测的操作几何**。共享策略预测腕/末端 \(p_0\) 与抓取中心 \(p_1\) 的相机系轨迹；几何条件翻译器再结合 \(T_{\mathrm{base}\leftarrow\mathrm{cam}}\)、Jacobian 与本体状态产出可执行命令。人手当独立 embodiment 直接监督共享几何，**不必**先做 robot-video inpainting 或显式 retargeting。
- **对 wiki 的映射：**
  - [UCAG-P 论文实体](../../wiki/entities/paper-ucag-p.md)
  - [VLA](../../wiki/methods/vla.md)
  - [Qwen-RobotManip](../../wiki/entities/qwen-robot-manip.md)（文内对照：相机系 EEF 仍绑机器人末端）

### 2) 预训练语料（§3 / Table 1–2）

- **核心贡献：** 11 个子集、**1,020,672** episode、**6,373.586 h**、约 **9** 种 embodiment。真机 RoboChallenge / RoboCoin / DROID **266.3 h**；仿真 InternData / RoboCasa GR-1 / LIBERO / RoboTwin **3,767.5 h**（InternData-MultiRobot 占 57.26%）；人手 VITRA / EgoDex / EgoVerse **2,339.7 h**（36.71%，伪动作）。另加 ShareRobot / RefSpatial-v2 / RoboVQA / RoboAfford 等 VL 样本（不计入小时）。缺失标签用 mask，不合成假监督。
- **对 wiki 的映射：**
  - [Open X-Embodiment](../../wiki/concepts/open-x-embodiment.md)
  - [EgoVerse](../../wiki/entities/paper-egoverse.md)

### 3) 相机系动作空间与架构（§4）

- **核心贡献：** 每步几何块 \(g\in\mathbb{R}^{30}\)：左右臂各 10 维（\(\Delta p_0,\Delta p_1,\sin/\cos\) 平面转、夹爪、padding）+ 相机运动 10 维。翻译器输出 **80 维稀疏命令**（左右臂/EE/手、腰、底座各 10 维槽）。骨干 **Qwen3-VL-4B-Instruct** + 可学习 action-query；motion head 为残差 MLP；action head 为 2 层 8 头 Transformer，融合运动、相机外参、Jacobian 与 8 个 query 池化的 VLM 隐状态。Horizon **H=30**。损失为掩码 L1：\(\mathcal{L}=\mathcal{L}_{geo}+\lambda_{cmd}\mathcal{L}_{cmd}\)。
- **对 wiki 的映射：**
  - [UCAG-P 官方仓归档](../repos/ucag-p.md)
  - [UHAS](../../wiki/methods/uhas-unified-hand-action-space.md)（另一条「统一动作空间」：球面形变 vs 相机锚点）

### 4) 三阶段训练（§4.4 / Table 8.1）

- **Stage 1** 相机系特化：VLM + motion head，全有几何标签数据，**128×H20、200K steps**。
- **Stage 2** 几何→命令：仅翻译器，**GT 轨迹** + 标定/Jacobian/可执行标签，**8×H20、10K**（无图像）。
- **Stage 3** 人–机联合：策略预测轨迹进翻译器以匹配推理，**64×H20、10K**；去掉辅助 VL。AdamW；VLM lr \(1\times10^{-5}\)，动作模块 \(1\times10^{-4}\)。
- **对 wiki 的映射：**
  - [Xiaomi-Robotics-0](../../wiki/entities/xiaomi-robotics-0.md)（同实验室 Qwen3-VL-4B 族，侧重异步 chunk 而非统一几何）

### 5) 实验与局限（§5–6 / §9 / §11）

- **单 checkpoint、无 per-benchmark 微调：** LIBERO **98.3%**（Spatial/Object/Goal/Long：98.8 / 98.6 / 99.2 / 96.4）；RoboTwin Easy/Hard **88.7% / 89.2%**；RoboCasa GR-1 **62.0%**（相对 Qwen-VLA-Instruct **+0.4 / +2.6 / +2.0 / +5.3 pt**）。LIBERO-Plus **零样本 82.0%**（对 robot/light/bkg 强，相机/噪声/布局弱）。ALOHA→ARX 零样本 **35.0%**（源机 Easy 88.66%）。真机 Piper、每任务 100 条演示 / 20 次闭环：面包抓取 **60% vs π₀.₅ 20%**（人→机）、开抽屉 **90% vs 85%**、叠碗 **75% vs 65%**。
- **短板：** OpenMicrowave 仅 11%/13%；接触丰富/铰接任务与跨形态迁移仍难；几何依赖标定、深度、运动学与 MediaPipe 关键点。
- **对 wiki 的映射：**
  - [Qwen-VLA](../../wiki/entities/qwen-vla.md)
  - [LIBERO](../../wiki/entities/libero-benchmark.md)
  - [DyPES-VLA](../../wiki/entities/paper-dypes-vla.md)（对照：不统一动作格式，改共享动力学 + MoE 原生头）
  - [跨具身迁移知识链](../../wiki/overview/hub-cross-embodiment.md)
  - [跨具身策略迁移选型指南](../../wiki/queries/cross-embodiment-transfer-strategy.md)
  - [WAM / VLA / 跨本体 9 篇技术地图](../../wiki/overview/wam-vla-cross-embodiment-9-papers-technology-map.md)

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 关联 wiki 页面的参考来源段落已添加 ingest 链接
