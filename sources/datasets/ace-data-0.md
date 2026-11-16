# ACE-Data-0（Hugging Face 数据集）

> 来源归档（dataset）

- **标题：** ACE-Data-0
- **类型：** dataset / multimodal / HOI / HSI
- **Hugging Face：** <https://huggingface.co/datasets/ACERobotics/ACE-Data-0>
- **组织：** ACERobotics（大晓机器人 Ace Robotics）
- **论文：** <https://arxiv.org/abs/2607.28625>
- **项目页：** <https://ace-data-engine.github.io/ACE-Data-0/>
- **许可：** `ace-data-0-research-license`（**非商业学术研究**，禁止再分发、再识别、生物识别/监控；HF README 声明 gated，但 HF API 截至 2026-10-14 实测 `gated: false`）
- **入库日期：** 2026-08-10（2026-11-14 核对 HF API JSON 与 README 全部 379 行）
- **核对范围：** 2026-11-14 读取 Hugging Face 数据集 API 元数据（`id`, `gated`, `usedStorage`, `lastModified`, `downloads`, `cardData`, `siblings`）及 `README.md` 全部 379 行（YAML frontmatter、Room-scale/Table-scale 目录布局与协议说明）。
- **通读：** 已完成
- **一句话说明：** 真实家居双尺度同步多模态人类活动集：ego/exo 视频、全身与手部运动、物体 6-DoF、音频与全掌触觉；截至 2026-10-14 已在 HF 上传约 `22.18 TB`、`≥100,000` 文件。

## 访问、门控口径差异与文件结构（2026-11-14 核实）

| 项 | 内容 |
|----|------|
| **HF API 实测状态** | `lastModified: 2026-10-14`、`gated: false`、`private: false`、`usedStorage: 22,176,330,852,720 bytes`（约 **22.18 TB**）、`downloads: 112,482`；`siblings` 返回达 100,000 条 API 上限，可见 `.gitattributes`、`LICENSE`、`README.md` 及 `Room-scale/`（含 `body_shape/`、`calibration/`、`date_to_take_mapping_20261006.csv`、`take-000000` 等） |
| **README 文本与现状差异** | `README.md` YAML 与正文仍保留 `extra_gated_*` 申请表、`Data_Files-Coming_Soon` 徽章及「Data files are not published yet」「This repository is gated」等过期说明；实际目录与文件命名也与 README 大小写略有差异（如 `Room-scale/body_shape/` 与 `take-xxxxxx` 为小写）；`Table-scale/`（LeRobot 兼容 parquet + `ACE-Data-Tactile-{0..10}`）因 API `siblings` 10 万条截断未独立枚举验证 |
| **用途限制（LICENSE）** | **仅非商业学术研究**；禁止再分发、再识别、生物识别/监控用途；个人条款不可转让 |
| **商业许可** | README 要求走 Community 另开线程 |
| **数据形态** | 含可识别人类参与者录像；需知情同意释放后的研究用途 |

## 规模（论文 / 项目页口径）

| 字段 | 数值 |
|------|------|
| 时长 | **150** 小时 |
| 帧数 | **17M** |
| 交互片段 | **75,000** episodes |
| 任务类 | **200** |
| 参与者 / 场景 | **50** 人 · **2** 家居环境 |
| 外视相机 | 典型 **≥8** 同步视角 |

## 关联资料

- 论文：[`sources/papers/ace_data_0_arxiv_2607_28625.md`](../papers/ace_data_0_arxiv_2607_28625.md)
- 项目页：[`sources/sites/ace-data-0-github-io.md`](../sites/ace-data-0-github-io.md)
- Wiki：[`wiki/entities/paper-ace-data-0.md`](../../wiki/entities/paper-ace-data-0.md)
