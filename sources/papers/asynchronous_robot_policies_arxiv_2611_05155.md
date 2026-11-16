# Asynchronous Robot Policies（ARP，arXiv:2611.05155v1）

> 来源归档（paper）；对应唯一实体：[ARP 异步机器人策略](../../wiki/entities/paper-asynchronous-robot-policies.md)

- **标题：** Asynchronous Robot Policies / 异步机器人策略
- **作者：** Sombit Dey、Luc Van Gool、Danda Pani Paudel（INSAIT）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.05155>
- **HTML：** <https://arxiv.org/html/2611.05155v1>
- **PDF：** <https://arxiv.org/pdf/2611.05155v1>
- **论文版本：** v1，2026-11-03
- **项目页：** <https://arp.insait.ai/>
- **入库日期：** 2026-11-05；本次原文复核：2026-11-16
- **线索来源：** [具身智能小站 2026-11-05 汇总](../blogs/wechat_embodied_station_43_papers_grasp_mesa_airoa_2026-11-05.md)；本次以 arXiv 原文复核，不沿用转述指标
- **核对范围：** arXiv HTML v1 全文：摘要、§1 引言、§2 相关工作、§3.1–3.4 方法、§4.1–4.5 评测/消融、§5 讨论与复现声明、附录 A（训练目标、前缀、延迟）/ B（扰动协议）/ C（动态任务）/ D（DROID/Franka/RoboLab）；并检查官方项目页的算法、实验、局限和 Code / Models 状态
- **通读：** 已完成

## 主要技术摘录（论文 v1）

1. **多频率双循环**：大 VLM 低频处理任务语义；视觉动作专家用最新图像、本体状态及最新已完成语义记忆高频生成动作块，避免串行等待 VLM。
2. **定宽语义接口**：从 VLM 中间层 KV 特征经 LN + 线性投影得到固定宽度记忆，通过 cross-attention 注入动作专家；记忆附有年龄信息。
3. **处理延迟错配**：训练时取早于当前观测的 VLM 记忆，默认混合分布为 20% 零延迟 + 80% 1–40 控制步随机年龄；AdaRMS 输入缓存年龄与执行延迟，动作前缀条件改善拼接。
4. **生成路径**：快速专家通过 flow matching 输出动作 chunk。论文仿真实现用 12 层 183M transformer、20 步动作块和 5 次 Euler 积分。

## 实验事实与限定

| 试验 | ARP | 对照 | 注意 |
|------|-----|------|------|
| LIBERO-Perturb，目标横移 5 cm | 51.5% | 9.5% | 同骨干/配方 π0 风格 1B 对照；论文 §4.3 |
| Dyn-Bump，必发生扰动 | 44.7% | 12.7% | 同骨干/配方 π0 风格 1B 对照 |
| LIBERO 标准平均 | 84.8%（1B） | 80.2%（π0 风格 1B） | 论文表 1；公开 π0 数据来源不同 |
| RoboLab 120 任务 / 960 回合 | 20.0%（ARP-3B） | 37.9%（π0.5-DROID） | 训练数据/视觉特征不同，不能归因单因素 |
| RoboLab 同一 ARP checkpoint 加 RTC | 37.5% | 31.9%（无 RTC） | 消融 +5.6pp；论文表 5 |

- 真机 Franka FR3、DROID 设置共 3 任务 × 10 次：蔬菜入碗、叠杯、马克笔入碗任务进度为 ARP 78/73/80%，π0.5-DROID 32/80/50%。叠杯上基线更好；进度不是完成率。
- VLM 从 5 Hz 降到 0.5 Hz 的测试展示多频率鲁棒性；但低于已测试频率的极限未知。
- 作者在 §4.5 表 4 明确说明：不做缓存陈旧训练时，在 0.5 Hz 下成功率只有 2.0%，而默认宽时延训练维持 81.8%；不能把收益归结为只用并行进程。
- 官方项目页和 arXiv v1 对部分实验的表述与展示存在口径差异（例如 LIBERO-Perturb 任务数），wiki 默认采用指定 arXiv v1，项目页特有数字单独标注。

## 开源状态与风险

- **待发布**：2026-11-16 核查官方项目页，显示 Code (soon)、Models (soon)，未列出可运行 GitHub 仓库或权重下载链接；因此没有可以核验的源码运行时序图。
- ARP 的动态反应优势不等同通用任务领先：RoboLab 综合指标低于 π0.5；真机实验范围小，异步相比同权重同步的部分额外收益未达显著。

## 对 wiki 的映射

- [paper-asynchronous-robot-policies](../../wiki/entities/paper-asynchronous-robot-policies.md) — 唯一论文实体，包含机制流程、训练、评测、局限
- [VLA 纵深路线 Stage 4](../../roadmap/depth-vla.md) — 异步部署与推理延迟问题
- [具身智能小站技术地图](../../wiki/overview/embodied-station-2026-11-05-43-papers-technology-map.md) — 保留既有期刊索引，避免重复建页
