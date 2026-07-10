---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: stub
updated: 2026-07-10
arxiv: "2510.03529"
related:
  - ./paper-humanoid-surgeon-in-vivo-laparoscopy.md
  - ../overview/paper-notebook-category-07-teleoperation.md
  - ../overview/humanoid-paper-notebooks-index.md
sources:
  - ../../sources/papers/humanoid_pnb_lapsurgie.md
summary: "现有手术机器人（如 da Vinci）昂贵、专用、只在高资源医疗中心普及，难以广泛部署。人形机器人能直接在为人设计的环境（含手术室）里工作、无需大改基础设施，是一条更可部署的路。LapSurgie 提出首个基于人形机器人的腹腔镜遥操作框架：让 G1 人形握住现成的手动腕式腹腔镜器械，核心是一套满足远心点（RCM）约束的逆映射策略，把操作者手部目标位姿解算成器械手柄该摆的姿态，从而精确控制未经改装的市售手术工具；再配一个带立体视觉实时反馈的控制台。14 人用户研究（peg-transfer）显示：在精度上人形系统可与 dVRK 金标准相当（新手 p=0.386，外科医生甚至更优），但完成时间明显更慢——证明了用人形机器人做微创手术遥操作的可行性，同时也暴露了速度瓶颈。"
---

# LapSurgie

**LapSurgie: Humanoid Robots Performing Surgery via Teleoperated Handheld Laparoscopy** 收录于 [Humanoid Robot Learning Paper Notebooks](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/index.html)（分类：07_Teleoperation），深读笔记已完成。本页为 **深读笔记索引实体**，正文要点编译自笔记；细节以笔记页与论文 PDF 为准。

## 一句话定义

现有手术机器人（如 da Vinci）昂贵、专用、只在高资源医疗中心普及，难以广泛部署。人形机器人能直接在为人设计的环境（含手术室）里工作、无需大改基础设施，是一条更可部署的路。LapSurgie 提出首个基于人形机器人的腹腔镜遥操作框架：让 G1 人形握住现成的手动腕式腹腔镜器械，核心是一套满足远心点（RCM）约束的逆映射策略，把操作者手部目标位姿解算成器械手柄该摆的姿态，从而精确控制未经改装的市售手术工具；再配一个带立体视觉实时反馈的控制台。14 人用户研究（peg-transfer）显示：在精度上人形系统可与 dVRK 金标准相当（新手 p=0.386，外科医生甚至更优），但完成时间明显更慢——证明了用人形机器人做微创手术遥操作的可行性，同时也暴露了速度瓶颈。

## 英文缩写速查

| 缩写 | 全称 | 解释 |
|---|---|---|
| RCM | Remote Center-of-Motion | 远心点约束：器械必须绕固定的切口点转动，腹腔镜手术的核心力学约束 |
| MIS | Minimally Invasive Surgery | 微创手术 |
| dVRK | da Vinci Research Kit | da Vinci 手术机器人研究套件，本文作金标准对比 + 提供 MTM 主手 |
| MTM | Master Tool Manipulator | dVRK 的主手操作器，作遥操作输入手柄 |
| HMD | Head-Mounted Display | 头戴显示（GOOVIS G3 Max），呈现立体内窥镜画面 |
| TRF | Trust-Region Reflective | 信赖域反射算法，用于求解逆映射的非线性最小二乘 |

## 为什么重要

- **「通用人形 = 可部署手术平台」**：核心论点是**部署性**——人形能直接进现成手术室、操作现成器械，绕开专用手术机器人的成本与基础设施壁垒。
- **逆映射是关键**：把被动器械显式建模 + RCM 几何约束，比硬 IK 更能稳定驱动手动腕式工具，思路可迁移到其他「人形操作现成人类工具」场景。
- **速度是主要短板**：完成时间约为 dVRK 的 2 倍，受控制方案复杂度与系统延迟拖累，是落地前必须攻克的瓶颈。
- **限制**：① 仅 14 人、以新手为主，泛化性待验证；② 仅 peg-transfer 单一受控任务，未测真实复杂术式；③ 从未真实临床部署；④ 逆映射需器械专属运动学参数，灵活性受限；⑤ 硬件设计、控制效率、几何建模精度均需提升。

## 解决什么问题

1. **手术机器人太贵太专用**：da Vinci 等平台成本高、依赖专用基础设施，只在高资源医院普及，难规模化下沉。 2. **能否用「通用人形」替代专用手术机器人？** 人形机器人可直接在为人类设计的手术室里操作现成器械，部署门槛低——但此前从未有人让人形执行腹腔镜手术。 3. **关键技术难点**：手动腕式腹腔镜器械是**被动运动链**，且必须满足**远心点（RCM）约束**（绕切口点转动）。如何把操作者的手部意图精确映射到这种无主动驱动、带 RCM 约束的器械上？

**目标**：一套**免改造现成器械、满足 RCM、带实时立体反馈**的人形腹腔镜遥操作框架，并用用户研究验证可行性。

## 核心机制

1. **首个人形腹腔镜遥操作框架**：证明通用人形机器人可在不改基础设施的前提下执行微创手术遥操作，开辟「手术机器人去专用化」的新路。
2. **RCM 约束下的逆映射重定向**：把被动腕式器械建模为运动链，闭式 + 优化求解手柄位形，精确驱动**未改装的现成手术工具**。
3. **沉浸式立体控制台**：dVRK MTM + 双目内窥镜 + 头戴立体显示，提供接近临床的遥操作体验。
4. **系统化用户研究**：14 名受试（含 2 名外科医生）在标准化 peg-transfer 上与手动、dVRK 金标准三方对比。

方法拆解（深读笔记小节）：逆映射策略（核心）；远心点（RCM）约束；硬件构成。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 07_Teleoperation |
| 深读笔记 | <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/LapSurgie__Humanoid_Robots_Performing_Surgery_via_Teleoperated_Handheld_Laparoscopy/LapSurgie__Humanoid_Robots_Performing_Surgery_via_Teleoperated_Handheld_Laparoscopy.html> |
| arXiv | <https://arxiv.org/abs/2510.03529> |
| 机构 | UC San Diego（ARCLAB，Michael C. Yip 组）等 |
| 作者 | Zekai Liang, Xiao Liang, Soofiyan Atar, Sreyan Das, Zoe Chiu, Peihan Zhang, Calvin Joyce, Florian Richter, Shanglei Liu, Michael C. Yip |
| 发表 | 2025-10-03（arXiv v1；v2 2026-02-16） |
| 项目主页 | [UCSD ARCLAB 论文页](https://ucsdarclab.com/autopublication/lapsurgie-humanoid-robots-performing-surgery-via-teleoperated-handheld-laparoscopy/) |
| 源码 | 截至当前未见公开代码仓库（以 arXiv 后续版本 / 项目页为准） |
| 笔记阅读日期 | 2026-06-23 |

## 实验与评测

- 本页为 **深读笔记编译** 的索引级摘要；量化 benchmark、消融与实机指标以 **深读笔记与论文 PDF** 为准（链接见 [参考来源](#参考来源)）。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-07-teleoperation](../overview/paper-notebook-category-07-teleoperation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)
- **活体升格：** 同团队 [Humanoid Surgeon（Nature 2026）](./paper-humanoid-surgeon-in-vivo-laparoscopy.md) 将 LapSurgie 框架推进至 **in vivo 猪模型腹腔镜胆囊切除术** 系统评估

## 参考来源

- [humanoid_pnb_lapsurgie.md](../../sources/papers/humanoid_pnb_lapsurgie.md)
- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/LapSurgie__Humanoid_Robots_Performing_Surgery_via_Teleoperated_Handheld_Laparoscopy/LapSurgie__Humanoid_Robots_Performing_Surgery_via_Teleoperated_Handheld_Laparoscopy.html>
- 论文：<https://arxiv.org/abs/2510.03529>

## 推荐继续阅读

- [机器人论文阅读笔记：LapSurgie](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/07_Teleoperation/LapSurgie__Humanoid_Robots_Performing_Surgery_via_Teleoperated_Handheld_Laparoscopy/LapSurgie__Humanoid_Robots_Performing_Surgery_via_Teleoperated_Handheld_Laparoscopy.html)
