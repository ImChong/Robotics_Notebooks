# HiFi-UMI-2K（Hugging Face）

> 来源归档（ingest 配套数据集）

- **标题：** HiFi-UMI-2K: High-Fidelity Robot-Free Manipulation Data
- **类型：** dataset
- **链接：** <https://huggingface.co/datasets/simple-world-lab/HiFi-UMI-2K>
- **论文：** [arXiv:2607.25895](https://arxiv.org/abs/2607.25895)
- **项目页：** <https://cloud.simpleai.tech/simple-world-lab/hifi-umi/>
- **机构：** 简易人工智能（Simple AI）/ Simple World Lab
- **许可：** CC BY 4.0
- **入库日期：** 2026-07-30
- **最后更新：** 2026-08-04
- **一句话说明：** 2000 小时高保真无机器人双臂操作示范（LeRobot v3 风格 Parquet+MP4）；源语料 >20k h；约 3 mm EE 精度、<40 µs 同步、六视角。

## 规模速查（HF card / 论文）

| 项 | 数值 |
|----|------|
| 公开时长 | **2,000 h** |
| 源语料 | **>20,000 h**，4.32M+ episodes，480+ scenes |
| 视角 | **6** 同步相机 / episode |
| EE 误差 | **~3 mm**（工作空间局部） |
| 同步 | **<40 µs** |
| 丢帧 | **<2 / h** |
| 夹爪角误差 | **<0.1°** |
| 格式 | LeRobot v3-style：`chunk-*/part-*/data/chunk-*/*.parquet` + MP4 |
| 轨迹重建 / WBC 回放 | 各约 **98%** 成功 |
| 许可 | CC BY 4.0 |

## 内容组成

每 episode 含：同步多视角视频、标定双臂 EE 轨迹、夹爪状态、语言标注、任务元数据、质控信息与归一化统计。

## 训练接口（HF card）

| 项 | 说明 |
|----|------|
| state / action | 各 **20** 维 = 右 10d + 左 10d；每手 `xyz(m) + rot6d(6) + gripper(rad)` |
| action 语义 | **绝对 next-state 目标**（非默认相对动作；论文实现可再转换） |
| 有效帧 | 保留 `valid.frame=false` 以对齐时间戳；训练过滤为 `true` |
| 视频 key | `head_main`、`head_main_stereo_right`、`{left,right}_hand_{up,down}` |
| 世界系 | 单录制内头/双手共世界系；**原点任意**（跨 episode 勿直接比绝对位姿）；+Z≈重力 |

## 对 wiki 的映射

- [HiFi-UMI 论文实体](../../wiki/entities/paper-hifi-umi.md)
- [论文归档](../papers/hifi_umi_arxiv_2607_25895.md)
- [HandUMI](../../wiki/entities/handumi.md)（同族 robot-free UMI，开源硬件对照）
- [具身数据金字塔](../../wiki/entities/paper-data-pyramid-embodied-manipulation.md)（UMI 层代表数据集）
