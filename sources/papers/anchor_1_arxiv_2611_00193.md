# Anchor 1（arXiv:2611.00193）

> 来源归档（paper / arXiv v1）

- **标题：** Anchor 1: Precise and Stable Humanoid Control for Whole-Body Manipulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.00193>（v1 提交 2026-10-30）
- **全文：** <https://arxiv.org/html/2611.00193v1>
- **PDF：** <https://arxiv.org/pdf/2611.00193>
- **项目页：** <https://archon.tech/research/anchor1>
- **作者与机构：** Archon Robotics（25 名作者：Caojun Wang, Checheng Yu, Gangcheng Jiang, Guangjun Zeng, Haoran Jiang, Hongyang Li 等）
- **入库日期：** 2026-11-03
- **再核日期：** 2026-11-21
- **核对范围：** 通读 `https://arxiv.org/html/2611.00193v1` 全文：Abstract、§1 Introduction、§2 Results（2.1–2.5，含 Table 1 与 Table 4）、§3 Discussion（3.1–3.3，含 Limitations）、§4 Materials and methods（4.1–4.5 稀疏参考标定、解耦误差反馈与双接口部署）。边界：未运行模型或硬件实验。
- **通读：** 已完成
- **线索来源：** [机器人不仅要完成任务，还要知道哪里出了错](../blogs/wechat_embodied_station_49_papers_sacq_action_aware_quantization_2026-11-03.md)（具身智能小站，2026-11-03）
- **一句话说明：** Anchor 1 通过 T-pose 几何标定把人类动作映射为骨盆/双腕/双踝五点稀疏参考（无需在线关节重定向），训练期配对离线筛选的机器人动作塑造全身协调分布，并将显式跟踪误差反馈与运动编码解耦，同时支持稀疏参考与 motion token 双接口。

## 开源状态

- **待发布**（2026-11-21 复核）：项目页 GitHub 与 Hugging Face 按钮均标注 "Available soon. Model release is expected in late 2026"；`ArchonRobotics` GitHub 组织尚无 Anchor 1 代码仓。页内提及的 `EGalahad/mimic-lite` 为对比基线，`bones-studio/seed` 为 Public track 外部数据源之一。

## 核心机制与实验要点（原文核对）

- **稀疏五点参考与分布策展（§4.1, §4.5.1）：** 基于人体与机器人 T-pose 标定分段尺度与坐标偏置形成 $H'$，提取骨盆、双腕、双踝五点位姿 $H'_5$；训练期配对离线重定向机器人参考 $R$ 并做质量筛选，共 406,222 条轨迹（1,081.4 小时，含镜像增强，其中自采 835.7 小时、公开 245.7 小时）。部署时直接使用 $H'_5$，免除在线关节重定向误差与延迟。
- **解耦误差反馈与双推理模式（§4.5.2–4.5.4）：** 编码器 $E$ + FSQ 量化 $Q$ 仅编码 `[0, 20, 40, 60] ms`（0.06 s 前瞻）稀疏目标窗生成 $z_t$；轨迹头 $K$ 重建近期目标（并辅以未来预测及循环一致性损失）；控制器 $D(z_t, o_t, e_t)$ 独立接收腕/踝 FK 跟踪误差 $e_t$。支持两类模式：直接输稀疏参考（$e_t$ 来自输入参考）或直接输 motion token（$e_t$ 来自 $K$ 重建目标），并区分 Global（全局轨迹）与 Re-locate（消除全局漂移依赖、按上一帧根位置平移）推理模式。
- **三轨基准（§2.1–2.3）：**
  - In-house：15,342 条、3,247 任务；MuJoCo 下 Anchor 1 Global 模式 local/global SR 为 88.4% / 88.7%。
  - Public：2,230 条（来自 BONES-SEED、CIRCLE、OMOMO、AMASS）。
  - Reachability：3,210 条（站立、浅蹲、深蹲、交错站立、单膝跪五姿）。
  - 真机（Table 1）：Reachability 51 条站立触达上腕部全局位置误差 44.58 mm（MimicLite 53.59 mm、SONIC-G1 98.15 mm）；In-house 72 条上腕部全局位置误差 75.69 mm（MimicLite 105.74 mm），局部腕部位置误差 29.43 mm（MimicLite 25.63 mm）。
- **下游任务（§2.5, Table 4）：**
  - 遥操作采数吞吐：踢箱 43 条/h、拾拖鞋 62 条/h（SONIC-SMPL 为 9 / 41 条/h）。
  - 自主策略（微调 GR00T N1.7，20 次试验）：Anchor 1 (re-locate + Recon. feedback) 使用 motion token 接口时踢箱 SR 0.90 / Score 0.81，拾拖鞋 SR 0.45 / Score 0.64；若使用编码器误差反馈但直接喂入 motion token 则 SR 为 0.00。
- **原文局限（§3.2）：** 仅在单一机器人形态上评测；稀疏运动学目标未显式建模物体几何、接触力或抓取稳定性；token 重建误差仍会引入一定跟踪偏差。

## 对 wiki 的映射

- [paper-anchor-1](../../wiki/entities/paper-anchor-1.md)
- [动作感知量化、失败定位与可验证自主：具身智能小站 11-03 期 49 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-03-49-papers-technology-map.md)
