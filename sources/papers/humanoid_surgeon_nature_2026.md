# In vivo feasibility study of humanoid robots in surgery（Nature 2026）

> 来源归档（ingest）

- **标题：** In vivo feasibility study of humanoid robots in surgery
- **类型：** paper（期刊）
- **期刊：** Nature（2026-07-08 在线发表）
- **DOI：** <https://doi.org/10.1038/s41586-026-10796-x>
- **PDF（期刊版）：** <https://www.nature.com/articles/s41586-026-10796-x.pdf>
- **作者公开预印本（全文）：** <https://arxiv.org/html/2607.07972>（arXiv:2607.07972v1，2026-07-08）；<https://arxiv.org/pdf/2607.07972>
- **核对范围：** 已阅读 arXiv:2607.07972v1 全文：Introduction、Results（台架工作空间/精度、O-ring 与 FLS 两项干实验室研究、两例活体猪胆囊切除及问卷）、Discussion、Methods（系统、轨迹拟合、inverse mapping 与 RCM）、代码/数据开放声明；核对 Fig. 1–5、Table 1–5 及关键数值。Nature DOI/发表信息与项目页交叉核对；未将新闻稿代替原文。
- **通读：** 已完成
- **项目页：** <https://humanoid-surgeon.github.io/>
- **作者：** Zekai Liang\*、Nikita Thareja、Peihan Zhang、Calvin Joyce、Soofiyan Atar、Florian Richter、Garth Jacobsen、Shanglei Liu、Ryan Broderick、Michael Yip（\* 通讯作者）
- **机构：** 加州大学圣地亚哥分校（UCSD）— Jacobs School of Engineering（ECE / ARCLAB）+ Department of Surgery / Center for the Future of Surgery
- **入库日期：** 2026-07-09
- **一句话说明：** 首次系统评估当代人形机器人腹腔镜手术能力：基于 **通用器械** 的人形腹腔镜 **遥操作框架**（延续 LapSurgie 线），经 **台架表征 → 干实验室用户研究 → 活体猪模型** 三级验证；完成 **腹腔镜胆囊切除术**，与 da Vinci 等专用平台对比量化可行性与临床就绪度。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | [humanoid-surgeon.github.io](https://humanoid-surgeon.github.io/) | 摘要、Fig.1–4 系统/台架/活体实验图、演示视频、BibTeX |
| 新闻稿 | [UC San Diego Today](https://today.ucsd.edu/story/surgeons-use-teleoperated-humanoid-robots-to-perform-live-surgery-a-world-first) | 两台 Surgie 人形、人机/机机团队、手术时长与标定痛点 |
| 前序预印本 | [LapSurgie arXiv:2510.03529](https://arxiv.org/abs/2510.03529) | 人形手持腹腔镜遥操作框架与 inverse-mapping / RCM |
| 姊妹工作 | [Humanoids in Hospitals arXiv:2503.12725](https://arxiv.org/abs/2503.12725) | 同一团队医院场景人形替代体技术路线 |
| 实验室 | [UCSD ARCLAB](https://ucsdarclab.com/projects/humanoid-robots-for-medicine/) | Humanoid Robots for Medicine 长期项目页 |
| 代码 | [Zenodo: Laparoscopic Humanoid Code](https://zenodo.org/records/18023650) | 作者公开了 80.5 MB 的 `g1_teleoperate-lap_surgie.zip`（v1），项目页 Code 按钮仍未提供 GitHub；已核对归档文件目录，未实际运行或逐文件审阅实现 |

## 摘要级要点

- **问题：** 医疗人力短缺与护理需求上升；医院大量工作仍是 **具身** 的（移动、操作、与人共处）。专用手术机器人（如 **da Vinci**）精度成熟，但 **重（~1800 lb）、占地大、需改造手术室、部署贵**；尚不清楚 **通用人形** 能否满足微创手术的精度、控制与安全。
- **系统：** 人形腹腔镜 **遥操作框架**，使用 **off-the-shelf 通用腹腔镜器械**（需适配器握持）；控制台含 **立体视觉** 实时反馈（延续 LapSurgie 的 inverse-mapping + **RCM** 约束思路）。
- **平台昵称：** **Surgie** — 约 **5 ft / 60 lb**（新闻稿），相对专用手术系统更轻、可移动、可融入现有 OR 工作流。
- **评估管线：**
  1. **Benchtop** — 手术工作空间与指令–执行跟踪表征（Fig. 2）
  2. **Dry-lab user studies** — 覆盖不同手术经验水平操作员
  3. **In vivo porcine** — 活体猪 **腹腔镜胆囊切除术**（cholecystectomy）：布署、戳卡、控制台操作、牵拉、分离、夹闭、肝床胆囊摘除（活体实验见 Fig. 4；Fig. 5 为工具运动学建模）
- **活体实验配置（新闻稿）：**
  - **人机团队：** 一台 Surgie + 一名人类助手外科医生 → 成功胆囊切除
  - **机机团队：** 两台 Surgie 并排协作 → 成功胆囊切除
- **相对专用平台：** 作者称遥操作人形精度 **可与专用手术机器人系统相当**；但术中需 **多次重新标定**，**手术时间显著长于** 成熟专用系统（类比早期 da Vinci 腹腔镜胆囊切除从 ~6 h 降至 ~30 min 的演进）。
- **开放挑战：** 遥操作 **延迟**（远程社区部署）、标定稳定性、临床部署前安全与监管路径；长期愿景含 **自主手术助手**（取器械、术后清理等全身任务）。


## 论文全文通读复核（2026-11-16）

- **系统与方法（Fig. 1、Fig. 5 / Methods §4）：** 实验平台为 Unitree G1；MTM 主端、足踏离合、立体内窥镜与 ROS2 通信。头部相机检测 ArUco 标志以定位 trocar 的 RCM；对手动腕式器械建立被动扩展运动学链，以目标 tool-tip 位姿反求机器人腕端位姿。夹爪通过伺服驱动器械指环，非器械自带电动关节。
- **台架（§2.1、Fig. 2）：** 随 RCM 位置变化，机器人仅在部分位置能达到 dVRK 参考工作空间；主从端跟随延迟约 **156 ms**，控制循环约 **24 ms**（其中逆运动学约 **11 ms**）。直线轨迹正交 RMS 偏差 **1.30 ± 0.03 mm**；圆轨迹径向 RMS 偏差 **10.40 ± 1.32 mm**，因此不能把「毫米级直线追踪」等同于整体运动精度已达到专用平台。
- **干实验室（§2.2、Tables 1–3）：** O-ring 转移 **18 人**（12 名新手、6 名外科医生），人形加权错误 **4.53 ± 3.14**、dVRK **4.59 ± 2.53**，但人形用时 **74.67 ± 39.22 s**，dVRK **43.01 ± 15.25 s**。FLS peg transfer **13 名外科医生**（8 初级、5 高级），人形平衡评分 **85.39 ± 16.25**，手动 **70.47 ± 27.53**，da Vinci Xi **97.67 ± 1.54**；人形明显落后于专用商用系统。FLS 评分是论文针对超时任务定义的归一化平衡评分，不应误认作原始标准 300 秒评分。
- **活体实验（§2.3、Fig. 4、Table 4）：** **两例**猪胆囊切除均完成，无需转为传统腹腔镜或开放术；术者为控制台上的资深外科医生，主要床旁协助由人类完成。第二台人形只在第一例短暂持镜/牵拉，不能写成「两台人形独立完成全流程」。**胆囊管夹闭/胆囊动脉结扎由人工腹腔镜完成**，因缺少适配机器人夹闭器。两例主动控制台操作时间为 **56:15 / 31:59**，机器人重新部署 **8 / 4** 次；这些时间不等于包含重标定和停顿的手术总时长。第二例出现轻微胆汁溢出与肝床出血，经处理后完成。
- **讨论与边界（§3）：** 当前限制包括动态 RCM 漂移、长时间重部署/校准、操作工作空间与臂展、延迟/轨迹误差、散热及无菌流程；作者明确未证明人体临床安全性或准备就绪。动物实验问卷总体临床就绪度平均 **2.5/5**（4 名填答者，按岗位回答各自题目）。
- **代码可用性（§6.3）：** 原文给出公开代码 DOI <https://doi.org/10.5281/zenodo.18023650>；Zenodo 记录含 `g1_teleoperate-lap_surgie.zip`、ROS2/遥操作目录。此处仅确认官方文件可下载及目录存在，**尚未检查实际运行入口或复现**，不能据此编写源码运行时序图。

## 核心摘录（面向 wiki 编译）

### 1) 从专用手术机器人到人形通用人形平台（§Introduction / 摘要）

- **摘录要点：** 传统机器人手术依赖 **purpose-built** 平台（da Vinci 等）；人形形态可在 **人类设计环境** 中移动与操作，有望降低偏远/资源不足地区的部署门槛。
- **对 wiki 的映射：**
  - [Humanoid Surgeon 论文实体](../../wiki/entities/paper-humanoid-surgeon-in-vivo-laparoscopy.md) — 问题陈述与临床动机

### 2) 遥操作框架与系统总览（Fig. 1）

- **摘录要点：** 外科医生在控制台遥操作人形执行腹腔镜任务；系统含器械适配、立体视觉链路与指令跟踪；与 [LapSurgie](https://arxiv.org/abs/2510.03529) 的 **inverse-mapping + RCM** 手持器械控制一脉相承。
- **对 wiki 的映射：**
  - [Humanoid Surgeon 论文实体](../../wiki/entities/paper-humanoid-surgeon-in-vivo-laparoscopy.md) — 系统架构
  - [Teleoperation](../../wiki/tasks/teleoperation.md) — 手术场景遥操作

### 3) 台架与干实验室（Fig. 2–3）

- **摘录要点：** 量化工作空间覆盖与 **command-execution tracking**；多经验层级用户研究对比任务表现与临床就绪度指标。
- **对 wiki 的映射：**
  - [Humanoid Surgeon 论文实体](../../wiki/entities/paper-humanoid-surgeon-in-vivo-laparoscopy.md) — 评测分层

### 4) 活体猪胆囊切除术（Fig. 4–5）

- **摘录要点：** 完整 MIS 流程：port setup → retraction → dissection → clipping → gallbladder removal；证据级 **in vivo feasibility**，非仅仿真或尸体模型。
- **对 wiki 的映射：**
  - [Humanoid Surgeon 论文实体](../../wiki/entities/paper-humanoid-surgeon-in-vivo-laparoscopy.md) — 核心结果
  - [LapSurgie 计划实体](../../wiki/entities/paper-notebook-lapsurgie-humanoid-robots-performing-surgery-via.md) — 前序框架 → Nature 活体升格

### 5) 局限与路线图（新闻稿 + 摘要收束）

- **摘录要点：** 术中多次 recalibration；procedure time 长于成熟专用系统；远程部署需降 **latency**；人形除手术外可承担 OR 物流类全身任务。
- **对 wiki 的映射：**
  - [Humanoid Surgeon 论文实体](../../wiki/entities/paper-humanoid-surgeon-in-vivo-laparoscopy.md) — 常见误区与展望
  - [Humanoids in Hospitals 计划实体](../../wiki/entities/paper-notebook-humanoids-in-hospitals-a-technical-study-of-huma.md) — 医院通用人形路线

## 参考来源（原始）

- 项目页：<https://humanoid-surgeon.github.io/>
- 论文：<https://doi.org/10.1038/s41586-026-10796-x>
- UCSD 新闻：<https://today.ucsd.edu/story/surgeons-use-teleoperated-humanoid-robots-to-perform-live-surgery-a-world-first>
