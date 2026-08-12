# Ego-OSCAR: Egocentric Open source Stereo CAptuRe System（arXiv:2608.08285）

> 来源归档（ingest）

- **标题：** Ego-OSCAR: Egocentric Open source Stereo CAptuRe System
- **缩写 / 框架：** **Ego-OSCAR**（Egocentric Open source Stereo CAptuRe）；数据集对外名 **Stereo-550** / **Ego-OSCAR-550h**
- **类型：** paper / open-hardware / egocentric / stereo-inertial / data-collection / dataset / vla-pretraining-substrate
- **arXiv：** <https://arxiv.org/abs/2608.08285>（Submitted 2026-08-08；PDF：<https://arxiv.org/pdf/2608.08285>）
- **项目页 / 3D：** <https://www.fpvlabs.ai/ego-oscar/cap>（Stereo Cap 3D 可视化）
- **数据集：** <https://huggingface.co/datasets/fpvlabs/stereo-550>（gated + FPV Labs 定制许可）
- **硬件规格 PDF：** <https://drive.google.com/file/d/1ZMgKqFdM65cAtcaI2s7SHr3Z-DljlHbe/view>
- **作者：** Gunjan Paul、Senthil Palanisamy、Satpal Singh Rathore、Pratyush Kumar Patnaik、Shubhanshu Khatana、Abhishek Anand（通讯：abhishek@fpvlabs.ai；作者同等贡献）
- **机构：** 第一人称视觉实验室（FPV Labs）
- **入库日期：** 2026-08-12
- **一句话说明：** ~USD 200 BoM 的开源硬件头戴 **同步全局快门立体相机 + 6 轴 IMU + 嵌入式编码 + MCU 看门狗**，配套采集管线；用分布式贡献者网络验证 **~550 h/相机** 标定立体 ego 语料（稠密自由格式动作字幕 + 论文宣称的全库手重建），定位「可众包扩展的 egocentric 基底」而非 Aria 级单机保真。

## 开源状态（步骤 2.5，2026-08-12）

| 项 | 核查结论 |
|----|----------|
| **项目页** | [fpvlabs.ai/ego-oscar/cap](https://www.fpvlabs.ai/ego-oscar/cap) 提供可穿戴 Stereo Cap 的 3D 视图；HF card 链到论文、Hardware Spec（Google Drive）、3D View |
| **数据集** | **已发布（受控）** — [fpvlabs/stereo-550](https://huggingface.co/datasets/fpvlabs/stereo-550)；需申请 gated access；许可为 FPV Labs 定制 license（非 CC） |
| **硬件文档** | Hardware Spec PDF（Drive）已公开；BoM/装配细节以论文 Table 1 + PDF 为准 |
| **采集软件 / CAD / 固件仓** | 论文宣称「All hardware designs, software, and the dataset are open-sourced」；截至入库日 `fpv-labs` GitHub org 仅见 [stera-app](https://github.com/fpv-labs/stera-app) / [stera-sdk](https://github.com/fpv-labs/stera-sdk)（手机 Stera / MobileEgo 线），**未发现** 独立 `Ego-OSCAR` 采集/固件/CAD 公开仓 |
| **结论** | **部分开源**：数据（gated）+ 硬件规格/3D 可视化已可及；**完整可运行采集栈的 GitHub 入口待核实**。wiki 源码时序图按「数据侧可用 / 采集仓未列」处理 |

## 摘录 1：问题与主张（§1 / §7）

- **瓶颈迁移：** VLA / world model 数据需求上升；遥操作贵、仿真有 sim2real、robot farm 偏保守；**egocentric 人类视频** 是中间地带，但缺「可众包复制」的传感器基底。
- **两端夹击：** Ego4D 等多为单目 rolling-shutter、无硬同步惯性；Project Aria 等高保真闭源不可自由复制。现代 VLA 常条件于世界系相机位姿，单目难恢复度量尺度。
- **三点贡献：**（1）开源硬件（CAD/BoM/布线/装配文档）；（2）开源采集管线（录制 daemon、IMU、时间同步、看门狗固件）；（3）**Ego-OSCAR-550h / Stereo-550**：~550 h/相机标定立体 + 同步 IMU，附 **209,315** 自由格式动作段与（论文）全库 WiLoR 3D 手重建。
- **分工声明：** [EgoHumanoid](https://arxiv.org/abs/2602.10106) 证明 egocentric 人数据可转策略；本文只解决 **如何以 ~USD 200 硬同步立体惯性采数**，端到端策略增益列为 future work。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-ego-oscar.md`](../../wiki/entities/paper-ego-oscar.md)；挂到 [Ego 分类 01 数据采集](../../wiki/overview/ego-category-01-data-collection.md)、[Ego4D](../../wiki/entities/paper-ego4d.md)、[Teleoperation](../../wiki/tasks/teleoperation.md)（与 UMI/HiFi-UMI 的「采集基底」对照）。

## 摘录 2：硬件与采集管线（§3）

- **传感器：** Dexcin USB 立体（Omnivision 全局快门，硬同步，1280×720@30，FOV 126°，基线 42 mm，单 USB UVC）；ICM-20948 6 轴 IMU（论文采样 120 Hz，I2C 可换更高端）；Radxa Rock 5C（RK3588，硬解 MJPEG / 硬编 H.264）；Xiao ESP32-S3（SoE ISR 时钟桥接、LED UX、>2 s 无心跳看门狗）。
- **形态：** 总重 ~280 g，运动帽檐安装；USB-PD 移动电源约 5–6 h；BoM ~USD 200 / INR 19,100；全 COTS + 3D 打印壳，无定制 PCB。
- **管线：** 5 min 切片 MP4（约 12–14 GB/h）；ESP 合并 SoE 时间戳 + IMU → UART；离线对齐；第 60 次 ISR 蓝 LED 闪光锚定帧号；Kalibr Cam-IMU 残差约 **700 µs**。
- **设计选择：** 姿态估计走**离线批处理**，可穿戴端专注重吞吐与可靠性。

**对 wiki 的映射：** 实体页画硬件/管线流程图与 BoM 关键项；强调「硬同步立体 + 看门狗」相对 GoPro/手机采集的工程差异。

## 摘录 3：评测与数据集（§4–§6 / Appendix A）

| 设定 | 要点 |
|------|------|
| 立体几何 | 每会话标定；重投影误差 <0.03 px；整流后平均极线误差 0.4 px（13 台设备） |
| IMU | Allan：accel 噪声密度 \(3.64\times10^{-2}\) m/s²/√Hz；gyro 偏置不稳 \(9.68\times10^{-4}\) rad/s（消费级） |
| 深度 / VO | SGBM/RAFT-Stereo 可用；VINS-Fusion stereo-inertial **12/20** 短序列收敛（无 ATE/RPE；非轨迹精度声明） |
| 手覆盖 | WiLoR 检出率 **94%**（覆盖率，非几何精度） |
| 部署 | 6 个月、25 贡献者、13 台设备、40+ 室内环境；**96%** 可用会话率 |
| Stereo-550 | 1,462 sessions；209,315 段；460 verbs / 32,630 object phrases；IMU 覆盖 86.9%；厨房向为主、约 1/3 时长在烹饪洗碗外（缝纫等） |
| 局限 | **未做** 用本数据训策略的增益实验；无位姿 GT；地理/人群集中（印度）；消费级 IMU；仅观测无实时拒收坏会话 |

**对 wiki 的映射：** 结论区分「传感器/部署可用性」与「机器人策略价值未证明」；注明 HF 目录卡未列手重建文件时与论文附录可能不一致。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-ego-oscar.md`**。
- 新建 `sources/sites/fpvlabs-ego-oscar.md`、`sources/datasets/stereo-550.md`；注册机构 `fpv-labs`。
- 交叉更新 [ego-category-01-data-collection.md](../../wiki/overview/ego-category-01-data-collection.md)、[paper-ego4d.md](../../wiki/entities/paper-ego4d.md)、[teleoperation.md](../../wiki/tasks/teleoperation.md)、[paper-hifi-umi.md](../../wiki/entities/paper-hifi-umi.md)（头戴立体对照）。
