# Sharpa Robotics 官网（sharpa.com）

> 来源归档（ingest · 公司官网一手页面）；核对日期：2026-11-21。

- **链接：** <https://www.sharpa.com/>
- **相关官方页：** <https://www.sharpa.com/pages/about>、<https://www.sharpa.com/blogs/research>
- **公司：** Sharpa Pte. Ltd.（夏帕机器人）
- **类型：** site（公司官网与研究博客索引）
- **入库日期：** 2026-11-21
- **核对范围：** 完整通读 `https://www.sharpa.com/` 首页、`https://www.sharpa.com/pages/about` 关于页及 `https://www.sharpa.com/blogs/research` 研究文章目录页正文与产品/架构字段；未逐篇点击并声称通读站内全部新闻稿与外链论文。
- **通读：** 已完成
- **一句话说明：** Sharpa 官方门户展示全栈灵巧操作产品矩阵（D01、Wave W01/W02、AE01）与分层 VTLA 模型 CraftNet（System 2 / System 1 / System 0）及商业化现场应用。

## 官网核心摘录（2026-11-21 抓取）

### 产品矩阵

| 产品 | 类型 | 官网披露关键指标（公司自报） |
|------|------|------------------------------|
| **Droid D01** | 全身触觉人形操作平台 | 集成 Wave 灵巧手 + 7-DoF 臂 + 全身电子皮肤 + 原生遥操；臂负载自重比约 `1:1`、末端最高速度 `>10.5 m/s`、重复精度 `0.2 mm`、全身触觉力分辨率 `0.2 N` |
| **Wave W01** | 22-DoF 触觉灵巧手 | 官网研究页与 `/pages/about` 称为“22-DoF dexterous hand with tactile sensing”，被多篇合作论文用作物理平台 |
| **Wave W02** | 新一代轻量化灵巧手 | 21 主动 DoF、全手触觉传感、IP54 防尘防水 |
| **Avatar AE01** | 触觉外骨骼数据手套 | 22 DoF 原生孪生设计、每指尖 256 级动态触觉振动反馈、免标定、抗环境干扰 |

### 分层模型与系统架构（CraftNet VTLA）

官网 `Our Intelligence` 与 `/pages/about` 明确定义 Sharpa 专有端到端分层 **VTLA（Vision-Touch-Language-Action）** 模型 **CraftNet**：

- **System 2（Reasoning Brain，约 1 Hz，VLM）**：接收语言与视觉输入进行推理，输出 **Semantic Intent** 与 **Progress Vector**；
- **System 1（Motion Brain，约 10 Hz）**：生成 **Physical Intent** 与 **Coarse Actions**；
- **System 0（Interaction Brain，约 100 Hz）**：基于实时触觉与本体感受（力、力矩等）闭环生成 **Fine Actions**，并向高层回传 **State Vector**。

### 商业落地与研究目录

- **Sharpa × DQ**：`/pages/about` 称 Sharpa 机器人于 DQ 门店运行 55 步自主制餐流程、每日 12 小时营业。
- **Research Blog**：收录与高校/研究机构合作及自研的多项工作（含 SaTA、SimToolReal、EgoScale、Tacmap、DexEMG、MoDE-VLA、T-Rex、FTP-1、Do as I Do、TacBPM、UVTA、GALATEA、Dex-X、WM-Craftnet、UniLab 等）。
- **开源口径**：公司门户面向商用与开发者合作；未设全栈统一开源仓库，各学术工作按论文条目独立公布或暂未开源。

## 对 wiki 的映射

- [夏帕机器人（Sharpa Robotics）](../../wiki/entities/sharpa-robotics.md)
- [CraftNet](../../wiki/entities/craftnet.md)
- [Sharpa D01](../../wiki/entities/sharpa-d01-robot.md) / [Sharpa W02](../../wiki/entities/sharpa-w02-hand.md) / [Sharpa AE01](../../wiki/entities/sharpa-ae01-data-glove.md)
