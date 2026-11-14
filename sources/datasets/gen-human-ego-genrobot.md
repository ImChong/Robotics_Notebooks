# Gen-HumanEgo（GenRobot / Hugging Face）

> 来源归档（ingest 配套数据集）

- **标题：** Gen-HumanEgo
- **类型：** dataset / huggingface
- **链接：** <https://huggingface.co/datasets/genrobot2025/Gen-HumanEgo>
- **项目页：** <https://www.genrobot.ai/data/open-dataset>
- **机构：** 简智机器人（GenRobot / JZ Robot）
- **许可：** CC BY-SA 4.0
- **访问（2026-11-14 复核）：** Hugging Face 现标记 **Gated dataset**；需登录、填写用途申请，页面称提交后自动授权（2026-09-23 归档旧口径为 ungated，已过时）。
- **入库日期：** 2026-09-23
- **官方技术原文：** <https://www.genrobot.ai/blog/gen-human-ego>（2026-09-23）
- **官方数据结构文档：** <https://docs.genrobot.ai/guides/das-ego-data-introduction>
- **核查日期：** 2026-11-14
- **核对范围：** 已通读 Hugging Face 上公开可见的完整 Gen-HumanEgo 数据卡（Dataset at a glance、Repository structure、What's inside an episode、Documentation and tools、Contact）；通读 GenRobot 2026-09-23 官方技术博客全文（多视角 DFM、HandTrack、Ego-Depth、三级语义、限制与后续方向）；通读官方 DAS-Ego Data Introduction 数据文档（topic/频率、设备 ID、坐标系/21 点手骨架、示例及解析入口）。数据集现为 gated，未登录下载受控的 MCAP 数据文件及原始 README、未运行数据处理代码或验证传感器实测精度。
- **通读：** 已完成
- **一句话说明：** 简智机器人 **RealOmni-Open** 开放数据计划中的 **人类第一视角** 子集：约 **1,848 h / 44,632 episodes / 10,257 任务**，六相机 DAS-Ego 同步 RGB + 双手 3D/MANO + Ego-Depth + 三级层次语义标注，episode 以 **MCAP** 打包；官方 **DFM** 离线处理提供结构化监督。

---

## 规模（HF README，入库日快照）

| 指标 | 数值 |
|------|------|
| 总时长 | **1,847.7 h** |
| Episodes | **44,632** |
| 唯一任务 | **10,257** |
| 场景域 | Home、business、industry、agriculture |
| RGB 视角 | **6** 路同步（DAS-Ego） |
| 分辨率 | **1600 × 1300** |
| 帧率 | **30 FPS** |
| 文件格式 | **MCAP** |
| 模态 | 每手 **21** 3D 关键点、MANO、hand mesh、depth、video/task/subtask 标注 |
| Hub 体量标签 | `n>1T`（HF size_categories） |
| 许可 | **CC BY-SA 4.0** |

> **版本差异：** 2026-09-23 旧版开放门户曾宣传 RealOmni-Open 为 10Kh / 1M+ clips；2026-11-14 复核门户已列独立的 10Kh RealOmni-Open 条目（13K+ hours / 5M+ clips 的公司标称），不能据此改写本数据集 1,847.7 h / 44,632 episodes 的数据卡口径。Gen-HumanEgo 目前仍按 episode MCAP 组织，但下载需要申请访问。

## 目录结构（任务层级）

```text
Gen-HumanEgo/
├── README.md
├── egov4_urdf.zip
├── assets/
└── data/
    ├── industry/logistics/sorting_and_packing/daily_work/00/<uuid>.mcap
    ├── domestic_services/living_room/clothing_organization/iron_clothes/c0/<uuid>.mcap
    └── business/restaurant/bakery/daily_work/a9/<uuid>.mcap
```

末两级为 **两位前缀目录 + episode MCAP 文件名**。

## MCAP 内 topic（README 摘要）

| 数据 | 说明 | Topic |
|------|------|-------|
| Multi-view RGB | 第一人称多视角观测 | `/robot0/sensor/camera[0-6]/compressed` |
| Hand reconstruction | 3D 关键点、MANO、hand mesh、质量字段 | `/robot0/handtracking/left`, `/robot0/handtracking/right` |
| Ego-Depth | 大 FOV 工作空间深度 | `/robot0/sensor/camera2/depth` |
| Hierarchical annotations | video / task / subtask 三级语义与时间边界 | `/robot0/annotation_v2/` |

**annotation_v2** 三级：**Video**（整段描述）→ **Task**（连续片段目标 + 时间/场景）→ **Subtask**（细粒度动作 caption + `is_success` + 物体/属性/空间关系）。

## 原始技术机制与验证边界（2026-11-14 全文复核）

- **多视角 DFM：** 官方技术博客描述以六路同步视频、相机内外参及时间连续性约束进行跨视角特征融合；稀疏 hand queries 经专门重建头输出每手 21 个三维点、MANO 参数和 mesh。这是对原始 RGB 的**离线推断标注**，不能视为每一帧都由外部动作捕捉设备提供真实手部位姿。
- **手部评估：** 官方称采用带标记的独立多相机 mocap 装置作高精度参考，在其评测设置下达到 sub-centimeter 误差；没有公开可复现实验表格与受试分布，不能解读为所有公开视频始终保持同精度。
- **Ego-Depth：** 博客称采用跨视角注意力扩大单图可用深度视场；公开的 Full-FOV 评测规格为 157°×132°±5°、1 m 深度误差 ≤±4%、≥99.5% fill rate。为公司在指定设置下的性能声明，非已实测每个 episode 的保证。
- **语义：** 先自然录制、后分层标注；video/task/subtask 各级带时间段，subtask 可含 `is_success`，用于保留重试/失败的上下文。博客中的整理沙发例子按 0–168.5 s task 与多段 subtask 展示层次语义。
- **Topic/坐标：** 官方 DAS-Ego 文档实际列明 `camera0` 到 `camera5` 六路压缩 RGB（30 Hz）、`/robot0/sensor/imu`（200 Hz）、`/robot0/handtracking/left/right`（30 Hz）、`/robot0/vio/eef_pose`（30 Hz）及帧/时间段有效性字段；手部 `boneData[i].toGlobal.position` 使用基于初始 Ego 位姿建立的 world frame。
- **可访问性：** 2026-11-14 官方 HF UI 明确显示 `Gated dataset` 与表单（提交后自动授权）。公开数据卡全文可读，不代表受控 MCAP 文件可匿名直接下载。未对实际 MCAP 样本实施端到端核验。

## 文档与工具（项目页 / HF 交叉核查）

| 资源 | 链接 | 开放程度 |
|------|------|----------|
| DAS-Ego 数据说明 | <https://docs.genrobot.ai/guides/das-ego-data-introduction> | 在线文档 |
| MCAP 浏览器可视化 | <https://monitor.genrobot.click/#/index> | Web 工具 |
| DAS DataKit | <https://github.com/genrobot-ai/das-datakit> | **已开源** |
| Ego URDF | Hub 内 `egov4_urdf.zip` | 随数据集发布 |
| DAS-Ego 硬件 | <https://www.genrobot.ai/products/ego> | 产品页 |
| 官方技术博客 | <https://www.genrobot.ai/blog/gen-human-ego> | 一手技术解读，2026-09-23 |

## 快速下载（需申请数据访问权限）

1. 先登录 [Gen-HumanEgo 数据集页面](https://huggingface.co/datasets/genrobot2025/Gen-HumanEgo)，填写用途申请表；页面称提交即自动授权。
2. 获授权后以本人的 Hugging Face 凭据下载；数据量大，建议只拉取需要的 `data/<domain>/...` 子目录或单个 MCAP episode。
3. 不得把公开数据卡可读等同于匿名可下载；访问条款以 Hugging Face 页面当前要求为准。

## 对 wiki 的映射

- [gen-human-ego-dataset](../../wiki/entities/gen-human-ego-dataset.md)
- [genrobot-open-dataset.md](../sites/genrobot-open-dataset.md)
- [das-datakit.md](../repos/das-datakit.md) — MCAP 读取 / 可视化 / H5 转换工具链
- [cn-os-das-datakit](../../wiki/entities/cn-os-das-datakit.md)
