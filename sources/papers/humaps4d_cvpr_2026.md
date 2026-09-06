# HUMAPS-4D: A Multimodal Dataset for Human Motion Analysis with Physiological and Semantic Informations（CVPR 2026）

> 来源归档（ingest）

- **标题：** HUMAPS-4D: A Multimodal Dataset for HUman Motion Analysis with Physiological and Semantic informations
- **缩写：** **HUMAPS-4D**
- **类型：** paper / dataset / human-motion / biomechanics / multimodal
- **会议：** CVPR 2026（IEEE/CVF；Open Access pp. 21188–21197）
- **PDF：** <https://openaccess.thecvf.com/content/CVPR2026/papers/Dabrowski_HUMAPS-4D_A_Multimodal_Dataset_for_HUman_Motion_Analysis_with_Physiological_CVPR_2026_paper.pdf>
- **项目页：** <https://humaps4d.wp.imt.fr/>（归档见 [`sources/sites/humaps4d.md`](../sites/humaps4d.md)）
- **作者：** Matthieu Dabrowski、Ouala Ben Jemaa、Benjamin Allaert
- **机构：** IMT Nord Europe；Institut Mines-Télécom；里尔大学（Université de Lille）；Centre for Digital Systems, Lille, France
- **资助：** ANR France 2030（ANR-24-RRII-0002，Inria Quadrant Program）
- **入库日期：** 2026-09-06
- **一句话说明：** 首个在统一协议下融合外视 RGB、MoCap、IMU、sEMG、足底压力与三层语义语言标注的大规模人体 4D 运动数据集，面向隐私友好姿态推断与跨模态动作识别 benchmark。

## 开源状态（步骤 2.5）

- **项目页核查（2026-09-06）：** 提供 DUA 下载流程与传感器/标注说明；**无 GitHub 链**。
- **数据获取：** 签署 DUA 后邮件申请凭证。
- **结论：** **数据部分开放（学术 DUA）**；**官方代码未发布**。

## 摘录 1：规模与模态（论文 §3 + 项目页）

| 维度 | 内容 |
|------|------|
| 被试 | 32 名健康成人（18–42 岁） |
| 动作 | 30 类（静态 /  locomotion / dynamic / interaction） |
| 时长 | 14 h；320 段 × 2 min 30 s；每被试 10 session |
| 帧数 | >5.76M 同步帧；>6M 时间对齐图像 |
| MoCap | Qualisys 11×Miqus M3，120 Hz；42 marker + 26 joint + quaternion |
| sEMG | Del sys Trigno 16 电极，1259 Hz；各电极含 IMU 148 Hz |
| 足底压力 IPS | Moticon OpenGo 双鞋垫各 16 传感器，100 Hz |
| RGB | 3×720p @120 Hz；人脸模糊；内外参提供 |
| 语义 | 帧级时序分割 + 原子动作描述 + 临床风格运动评估叙事 |
| 人体测量 | 身高、性别、体重、下肢段长、足长等 |

## 摘录 2：Benchmark 任务族（论文 §4）

1. **Insole-based activity recognition（§4.1）** — 仅足底压力序列做 30 类动作识别；LOSO 协议；训练可用 RGB/MoCap/sEMG 作约束。融合模型全类平均 **90.33%**（Table 2），纯鞋垫 **82.71%**。
2. **3D pose estimation from foot pressure（§4.2）** — 由足底压力 + 加速度/力/压力中心推断 16 个上身/下肢关节 3D 位置（不含双臂）；4-fold CV。总体 MPJPE **31.1 cm**、Inconsistency **7.5 cm**、MPJAE **13.3°**（Table 3）。
3. **Expert-Level Semantic Analysis** — 附录给出语义运动分割等实验；论文称 2026 年将启动公开 challenge。

## 摘录 3：与 prior 数据集对比（Table 1 要点）

相对 SolePoser / P2P-Insole / MMVP 等视觉驱动姿态集，以及 MovePort / Smart-insole 等生物力学集，HUMAPS-4D 在 **被试数、动作多样性、RGB 视角、语义标注与人体测量** 上更完整，是少数同时覆盖 **vision + insole + sEMG + MoCap + 语义** 的统一协议资源。

**对 wiki 的映射：** [`wiki/entities/paper-humaps4d.md`](../../wiki/entities/paper-humaps4d.md)。

## 当前提炼状态

- [x] 论文摘要与规模表填写
- [x] 项目页开源核查
- [x] wiki 页面映射确认
