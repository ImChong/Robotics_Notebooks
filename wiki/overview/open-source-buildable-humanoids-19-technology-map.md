---
type: overview
tags: [overview, humanoid, open-source, hardware, 3d-printing, education, wechat]
status: complete
updated: 2026-11-02
related:
  - ../entities/roboprime.md
  - ../entities/zowi.md
  - ../entities/kayra.md
  - ../entities/humanoid-robot.md
  - ../entities/open-source-humanoid-hardware.md
  - ./robot-open-source-wechat-githubstore-18-projects.md
sources:
  - ../../sources/blogs/wechat_githubstore_19_buildable_open_source_humanoids_2026-11-02.md
summary: "GitHubStore 清单《19 款能自己造的开源人形机器人》的阅读坐标：按文章标价从 $50 到 $900 排序，新建 13 / 复用 6 个独立节点，并标出文章归类、价格与 README 不一致之处。"
---

# 能自己造的开源人形机器人 19 款：按成本与形态排序的阅读坐标

> **本页定位**：为 [19款能自己造的开源人形机器人](https://mp.weixin.qq.com/s/UUwUAcKdVrlo-N60Oar3Xg)（2026-11-02）提供阅读坐标；19 个项目均有唯一详情节点。价格为文章表述，未核实。

## 一句话观点

**这份清单把「能自己造」的机器人按成本从几十美元的 Arduino 玩具人形排到近千美元的 InMoov 上半身；其中数个项目（Bambot、Valera、Reachy Mini、XLeRobot）按 README 并非人形。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DOF | Degrees of Freedom | 自由度 |
| BOM | Bill of Materials | 物料清单 |
| STL | Stereolithography file | 3D 打印网格文件 |

## 为什么单独做这张地图

- 库内已有 6 个相关节点分散在各处，新增 13 个后需要一个统一入口；
- 文章的归类（人形）与若干项目的 README 自述不一致，需要显式标注。

## 项目一览

| # | 项目 | 价格（文章） | 节点 | 状态 |
|---|------|-------------|------|------|
| 1 | RoboPrime | 约 $60–100 | [roboprime](../entities/roboprime.md) | 新建 |
| 2 | Qmini | 价格未确认 | [qmini](../entities/qmini.md) | 复用 |
| 3 | Zowi | 约 €100 | [zowi](../entities/zowi.md) | 新建 |
| 4 | Otto Humanoid / Otto DIY | 约 $50 起 | [otto-diy-biped](../entities/otto-diy-biped.md) | 复用 |
| 5 | Kayra | 价格未确认 | [kayra](../entities/kayra.md) | 新建 |
| 6 | PLEN2 | 套件约 $499 | [plen2](../entities/plen2.md) | 新建 |
| 7 | Nova Desktop Humanoid | 约 $150 | [nova-desktop-humanoid](../entities/nova-desktop-humanoid.md) | 新建 |
| 8 | Bambot | 约 $300 | [bambot](../entities/bambot.md) | 新建 |
| 9 | Rapiro | 约 $390 | [rapiro](../entities/rapiro.md) | 新建 |
| 10 | Valera | 约 $400 | [valera](../entities/valera.md) | 新建 |
| 11 | Reachy Mini | 约 $399 | [reachy-mini](../entities/reachy-mini.md) | 新建 |
| 12 | Open Duck Mini | 低于 $400 | [open-duck-mini](../entities/open-duck-mini.md) | 复用 |
| 13 | MicroDuck | 约 $399 | [pollen-microduck](../entities/pollen-microduck.md) | 复用 |
| 14 | ROBOTIS MINI | 约 $540 | [robotis-mini](../entities/robotis-mini.md) | 新建 |
| 15 | Halley | 约 $500–800 | [halley-ambassador-robot](../entities/halley-ambassador-robot.md) | 新建 |
| 16 | Microban | 约 $550–600 | [microban-humanoid](../entities/microban-humanoid.md) | 复用 |
| 17 | Endeavor II | 约 $600 | [endeavor-ii-humanoid](../entities/endeavor-ii-humanoid.md) | 新建 |
| 18 | XLeRobot | 约 $660 | [xlerobot](../entities/xlerobot.md) | 新建 |
| 19 | InMoov | 上半身约 $900 | [inmoov-humanoid](../entities/inmoov-humanoid.md) | 复用 |

## 文章说法与核查的差异

| 项目 | 文章说法 | 核查 | 处理 |
|------|----------|------|------|
| Bambot | 归入「人形机器人」清单 | README 未称其为人形（仅称低成本 AI 机器人） | 按 README 描述，标注归类存疑 |
| Valera | 归入「人形机器人」清单 | README 称社交机器人，未称人形 | 同上 |
| Reachy Mini | 归入「人形机器人」清单 | README 称 expressive robot（桌面机器人） | 同上；库内 painode-169 仅是索引条目 |
| XLeRobot | 归入「人形机器人」清单 | 双臂移动机器人（SO-101 机械臂 + 移动底盘，据文章） | 同上 |
| Endeavor II | 价格约 $600 | README 为 $600–$800 | 采用 README 区间 |
| PLEN2 / Halley / RoboPrime / Zowi | 文章呈现为现成项目 | 仓库最近提交分别为 2018 / 2015 / 2017 / 2026（zowi 为个人源文件仓库，官方 zowiLibs 为 2016） | 标注停更时间 |
| Nova / Rapiro（官方）/ InMoov 官网 | 文章给官网或第三方站点 | 无 GitHub 官方仓库，未访问官网 | 只记文章表述，标未核实 |

## 关联页面

- [人形机器人总览](../entities/humanoid-robot.md)
- [开源人形硬件对比](../entities/open-source-humanoid-hardware.md)
- [GitHubStore 18 个开源项目（同账号上一篇）](./robot-open-source-wechat-githubstore-18-projects.md)

## 参考来源

- [wechat_githubstore_19_buildable_open_source_humanoids_2026-11-02.md](../../sources/blogs/wechat_githubstore_19_buildable_open_source_humanoids_2026-11-02.md)
