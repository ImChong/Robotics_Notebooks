# In vivo feasibility study of humanoid robots in surgery（Nature 2026）

> 来源归档（ingest）

- **标题：** In vivo feasibility study of humanoid robots in surgery
- **类型：** paper（期刊）
- **期刊：** Nature（2026-07-08 在线发表）
- **DOI：** <https://doi.org/10.1038/s41586-026-10796-x>
- **PDF：** <https://www.nature.com/articles/s41586-026-10796-x.pdf>
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
| 代码 | 项目页 Code 按钮暂为占位；Nature 参考文献含 Zenodo「Laparoscopic humanoid code」 | 待官方放出 GitHub 后补链 |

## 摘要级要点

- **问题：** 医疗人力短缺与护理需求上升；医院大量工作仍是 **具身** 的（移动、操作、与人共处）。专用手术机器人（如 **da Vinci**）精度成熟，但 **重（~1800 lb）、占地大、需改造手术室、部署贵**；尚不清楚 **通用人形** 能否满足微创手术的精度、控制与安全。
- **系统：** 人形腹腔镜 **遥操作框架**，使用 **off-the-shelf 通用腹腔镜器械**（需适配器握持）；控制台含 **立体视觉** 实时反馈（延续 LapSurgie 的 inverse-mapping + **RCM** 约束思路）。
- **平台昵称：** **Surgie** — 约 **5 ft / 60 lb**（新闻稿），相对专用手术系统更轻、可移动、可融入现有 OR 工作流。
- **评估管线：**
  1. **Benchtop** — 手术工作空间与指令–执行跟踪表征（Fig. 2）
  2. **Dry-lab user studies** — 覆盖不同手术经验水平操作员
  3. **In vivo porcine** — 活体猪 **腹腔镜胆囊切除术**（cholecystectomy）：布署、戳卡、控制台操作、牵拉、分离、夹闭、肝床胆囊摘除（Fig. 4–5）
- **活体实验配置（新闻稿）：**
  - **人机团队：** 一台 Surgie + 一名人类助手外科医生 → 成功胆囊切除
  - **机机团队：** 两台 Surgie 并排协作 → 成功胆囊切除
- **相对专用平台：** 作者称遥操作人形精度 **可与专用手术机器人系统相当**；但术中需 **多次重新标定**，**手术时间显著长于** 成熟专用系统（类比早期 da Vinci 腹腔镜胆囊切除从 ~6 h 降至 ~30 min 的演进）。
- **开放挑战：** 遥操作 **延迟**（远程社区部署）、标定稳定性、临床部署前安全与监管路径；长期愿景含 **自主手术助手**（取器械、术后清理等全身任务）。

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
