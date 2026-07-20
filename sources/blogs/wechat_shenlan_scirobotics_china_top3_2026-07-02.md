# 近五年《Science Robotics》中国顶尖高校机器人研究盘点

> 来源归档（blog / 微信公众号）

- **标题：** 近五年《Science Robotics》中国顶尖高校机器人研究盘点
- **类型：** blog
- **作者：** 深蓝AI（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/hbz9VPNH84CUtqORychPeA
- **发表日期：** 2026-07-02
- **入库日期：** 2026-07-20
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + `wechat-article-for-ai`（Camoufox）；`--no-images`；Jina Reader 对该链接触发微信 CAPTCHA，未采用
- **原始抓取落盘：** [`sources/raw/wechat_shenlan_scirobotics_china_top3_2026-07-02/`](../raw/wechat_shenlan_scirobotics_china_top3_2026-07-02/)
- **一句话说明：** 按 **浙大 / 北航 / 清华** Top-3 通讯作者单位，盘点 2022–2026 中国内地机构在 *Science Robotics* 上的代表性工作（微型集群、仿生软体、管内机器人、类脑芯片等），共 **9 篇** 论文升格为独立 wiki 实体页。

## 核心摘录（归纳，非全文）

### 选题边界

- **统计口径：** 2022–2026，中国内地机构作为 **通讯作者单位** 在 *Science Robotics* 发文；文内强调 Top-3 为 **浙江大学、北京航空航天大学、清华大学**。
- **非穷尽：** 作者声明受篇幅限制大量优秀工作未收录——本库按文内列出的 **9 篇** 各建独立节点，不把公众号本身当作唯一评价值排序。

### 三校代表作（9 篇）

| 机构 | 论文（文内标题） | DOI / arXiv | wiki 实体 |
|------|------------------|-------------|-----------|
| 浙大 | Swarm of micro flying robots in the wild | [10.1126/scirobotics.abm5954](https://doi.org/10.1126/scirobotics.abm5954) | [paper-swarm-micro-flying-robots-in-the-wild](../../wiki/entities/paper-swarm-micro-flying-robots-in-the-wild.md) |
| 浙大 | Bistable soft jumper capable of fast response and high takeoff velocity | [10.1126/scirobotics.adm8484](https://doi.org/10.1126/scirobotics.adm8484) | [paper-bistable-soft-jumper-magnetic](../../wiki/entities/paper-bistable-soft-jumper-magnetic.md) |
| 浙大 | Microsaccade-inspired Event Camera for Robotics | [10.1126/scirobotics.adj8124](https://doi.org/10.1126/scirobotics.adj8124) · [arXiv:2405.17769](https://arxiv.org/abs/2405.17769) | [paper-microsaccade-inspired-event-camera](../../wiki/entities/paper-microsaccade-inspired-event-camera.md) |
| 北航 | Octopus-inspired sensorized soft arm（E-SOAM） | [10.1126/scirobotics.adh7852](https://doi.org/10.1126/scirobotics.adh7852) | [paper-octopus-inspired-esoam-soft-arm](../../wiki/entities/paper-octopus-inspired-esoam-soft-arm.md) |
| 北航 | Aerial-aquatic robots … hitchhiking on surfaces | [10.1126/scirobotics.abm6695](https://doi.org/10.1126/scirobotics.abm6695) | [paper-aerial-aquatic-remora-hitchhiking-robot](../../wiki/entities/paper-aerial-aquatic-remora-hitchhiking-robot.md) |
| 北航 | Miniature deep-sea morphable robot … | [10.1126/scirobotics.adp7821](https://doi.org/10.1126/scirobotics.adp7821) | [paper-miniature-deep-sea-morphable-robot](../../wiki/entities/paper-miniature-deep-sea-morphable-robot.md) |
| 清华 | Sub-centimeter pipeline inspection robot | [10.1126/scirobotics.abm8597](https://doi.org/10.1126/scirobotics.abm8597) | [paper-subcentimeter-pipeline-inspection-robot](../../wiki/entities/paper-subcentimeter-pipeline-inspection-robot.md) |
| 清华 | TianjicX neuromorphic chip … | [10.1126/scirobotics.abk2948](https://doi.org/10.1126/scirobotics.abk2948) | [paper-tianjicx-neuromorphic-chip-robots](../../wiki/entities/paper-tianjicx-neuromorphic-chip-robots.md) |
| 清华 | NeuroGPR brain-inspired multimodal place recognition | [10.1126/scirobotics.abm6996](https://doi.org/10.1126/scirobotics.abm6996) | [paper-neurogpr-brain-inspired-place-recognition](../../wiki/entities/paper-neurogpr-brain-inspired-place-recognition.md) |

### 文内收束趋势（编译）

1. **仿生 / 软体**：章鱼臂、鮣鱼吸盘、双稳态跳跃等，用柔性结构补刚性机器人局限。
2. **微型化与集群**：密林微型无人机蜂群、亚厘米管道机器人。
3. **底层算力**：TianjicX / NeuroGPR 等类脑芯片与多模态场所识别。

## 对 wiki 的映射

- 九篇均已升格为 `wiki/entities/paper-*.md`（见上表）；交叉枢纽含 [EGO-Planner Swarm](../../wiki/entities/ego-planner-swarm.md)、[locomotion](../../wiki/tasks/locomotion.md)、[manipulation](../../wiki/tasks/manipulation.md)、[teleoperation](../../wiki/tasks/teleoperation.md)、[quadruped-robot](../../wiki/entities/quadruped-robot.md) 等。
- 姊妹盘点（全球实验室视角，文内外链）：公众号前作「近五年 Science Robotics 全球顶尖实验室」——本条仅覆盖国内 Top-3。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 9 篇论文 DOI / 独立 wiki 实体
- [x] 开源状态按项目页 / Zenodo / 论文核查写入各实体页
