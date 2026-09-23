# Gen-HumanEgo（GenRobot / Hugging Face）

> 来源归档（ingest 配套数据集）

- **标题：** Gen-HumanEgo
- **类型：** dataset / huggingface
- **链接：** <https://huggingface.co/datasets/genrobot2025/Gen-HumanEgo>
- **项目页：** <https://www.genrobot.ai/data/open-dataset>
- **机构：** 简智机器人（GenRobot / JZ Robot）
- **许可：** CC BY-SA 4.0
- **访问：** **ungated**（入库日 2026-09-23）
- **入库日期：** 2026-09-23
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

> 官网 **RealOmni-Open DataSet** 叙事为 **10Kh / 1M+ clips** 全栈开放数据；**Gen-HumanEgo** 为其中 **人类 egocentric** 主线，当前经 HF 按 episode MCAP 分发。

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

## 文档与工具（项目页 / HF 交叉核查）

| 资源 | 链接 | 开放程度 |
|------|------|----------|
| DAS-Ego 数据说明 | <https://docs.genrobot.ai/guides/das-ego-data-introduction> | 在线文档 |
| MCAP 浏览器可视化 | <https://monitor.genrobot.click/#/index> | Web 工具 |
| DAS DataKit | <https://github.com/genrobot-ai/das-datakit> | **已开源** |
| Ego URDF | Hub 内 `egov4_urdf.zip` | 随数据集发布 |
| DAS-Ego 硬件 | <https://www.genrobot.ai/products/ego> | 产品页 |

## 快速下载

```bash
pip install -U huggingface_hub
huggingface-cli download genrobot2025/Gen-HumanEgo --repo-type dataset
```

可按 `data/<domain>/...` 子树增量拉取单个 MCAP episode。

## 对 wiki 的映射

- [gen-human-ego-dataset](../../wiki/entities/gen-human-ego-dataset.md)
- [genrobot-open-dataset.md](../sites/genrobot-open-dataset.md)
- [das-datakit.md](../repos/das-datakit.md) — MCAP 读取 / 可视化 / H5 转换工具链
- [cn-os-das-datakit](../../wiki/entities/cn-os-das-datakit.md)
