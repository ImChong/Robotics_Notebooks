# 19款能自己造的开源人形机器人，附完整开源地址

> 来源归档（blog / 微信公众号）

- **标题：** 19款能自己造的开源人形机器人，附完整开源地址
- **类型：** blog
- **作者：** 小G / GitHubStore（微信公众号；文末链接上一篇「想进入机器人领域吗？学习这18个开源项目就够了」）
- **原始链接：** https://mp.weixin.qq.com/s/UUwUAcKdVrlo-N60Oar3Xg
- **发表日期：** 2026-11-02
- **入库日期：** 2026-11-02
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信内置 UA），正文 4598 字符
- **一句话说明：** 按成本从低到高列 19 个能自己 3D 打印或组装的开源人形 / 类人形机器人，给出价格、核心配置与开源地址。
- **说明：** 19 个项目 → 19 个唯一节点：**新建 13、复用 6**。价格为文章表述，入库日未核实；GitHub 仓库经 `git ls-remote` 与浅克隆 README 核对，无 GitHub 仓库的项目只记文章表述。

## 项目 → 节点

| # | 项目 | 价格（文章） | 节点 | 状态 | 备注 |
|---|------|-------------|------|------|------|
| 01 | RoboPrime | 约 $60–100 | [roboprime](../../wiki/entities/roboprime.md) | 新建 |  |
| 02 | Qmini | 价格未确认 | [qmini](../../wiki/entities/qmini.md) | 复用 | 文章另列 RoboTamerSdk4Qmini 软件仓库（git ls-remote 可达，未建节点） |
| 03 | Zowi | 约 €100 | [zowi](../../wiki/entities/zowi.md) | 新建 |  |
| 04 | Otto Humanoid / Otto DIY | 约 $50 起 | [otto-diy-biped](../../wiki/entities/otto-diy-biped.md) | 复用 |  |
| 05 | Kayra | 价格未确认 | [kayra](../../wiki/entities/kayra.md) | 新建 |  |
| 06 | PLEN2 | 套件约 $499 | [plen2](../../wiki/entities/plen2.md) | 新建 |  |
| 07 | Nova Desktop Humanoid | 约 $150 | [nova-desktop-humanoid](../../wiki/entities/nova-desktop-humanoid.md) | 新建 |  |
| 08 | Bambot | 约 $300 | [bambot](../../wiki/entities/bambot.md) | 新建 |  |
| 09 | Rapiro | 约 $390 | [rapiro](../../wiki/entities/rapiro.md) | 新建 |  |
| 10 | Valera | 约 $400 | [valera](../../wiki/entities/valera.md) | 新建 |  |
| 11 | Reachy Mini | 约 $399 | [reachy-mini](../../wiki/entities/reachy-mini.md) | 新建 |  |
| 12 | Open Duck Mini | 低于 $400 | [open-duck-mini](../../wiki/entities/open-duck-mini.md) | 复用 | Runtime 仓库在库内另有 open-duck-mini-runtime 节点 |
| 13 | MicroDuck | 约 $399 | [pollen-microduck](../../wiki/entities/pollen-microduck.md) | 复用 | RL 训练仓库在库内另有 pollen-microduck-rl 节点 |
| 14 | ROBOTIS MINI | 约 $540 | [robotis-mini](../../wiki/entities/robotis-mini.md) | 新建 |  |
| 15 | Halley | 约 $500–800 | [halley-ambassador-robot](../../wiki/entities/halley-ambassador-robot.md) | 新建 |  |
| 16 | Microban | 约 $550–600 | [microban-humanoid](../../wiki/entities/microban-humanoid.md) | 复用 |  |
| 17 | Endeavor II | 约 $600 | [endeavor-ii-humanoid](../../wiki/entities/endeavor-ii-humanoid.md) | 新建 |  |
| 18 | XLeRobot | 约 $660 | [xlerobot](../../wiki/entities/xlerobot.md) | 新建 |  |
| 19 | InMoov | 上半身约 $900 | [inmoov-humanoid](../../wiki/entities/inmoov-humanoid.md) | 复用 |  |

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

## 对 wiki 的映射

- 阅读坐标：[能自己造的开源人形机器人 19 款：按成本与形态排序的阅读坐标](../../wiki/overview/open-source-buildable-humanoids-19-technology-map.md)
- 同账号前一篇 18 项清单见 [robot-open-source-wechat-githubstore-18-projects](../../wiki/overview/robot-open-source-wechat-githubstore-18-projects.md)。
