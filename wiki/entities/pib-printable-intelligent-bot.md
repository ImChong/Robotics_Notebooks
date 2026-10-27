---
type: entity
tags: [humanoid, education, open-source, 3d-printing, ros2, raspberry-pi]
status: complete
updated: 2026-10-27
project_id: pib-printable-intelligent-bot
code: https://github.com/pib-rocks/pib-backend
summary: "pib：德国 isento GmbH 的可完全 3D 打印开源人形，当前 pib#5（Raspberry Pi 5、ROS 2 Humble），AGPL-3.0，已用于超过 150 所学校。"
related:
  - ./humanoid-robot.md
  - ./open-source-humanoid-hardware.md
  - ./ros2-basics.md
  - ../overview/robot-open-source-wechat-githubstore-18-projects.md
sources:
  - ../../sources/blogs/wechat_githubstore_18_open_source_robot_projects.md
---

# pib（printable intelligent bot）

## 一句话定义

**pib（printable intelligent bot）** 由德国 isento GmbH 开发，是可完全 3D 打印的开源人形机器人，软件在 GitHub 组织 [pib-rocks](https://github.com/pib-rocks) 下发布（后端 [pib-backend](https://github.com/pib-rocks/pib-backend)，STL [pib-stls](https://github.com/pib-rocks/pib-stls)）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ROS 2 | Robot Operating System 2 | 机器人中间件 |
| PETG | Polyethylene Terephthalate Glycol | 推荐打印耗材 |
| AGPL | GNU Affero General Public License | 网络著佐权许可证 |
| BOM | Bill of Materials | 物料清单 |

## 为什么重要

- 教育场景落地规模较大的 3D 打印人形，软硬件文档完整。
- 在微信文章《想进入机器人领域吗？学习这18个开源项目就够了》中排第 **5** 位。

## 核心要点（据文章）

- 当前主推 pib#5：Raspberry Pi 5 机载计算，支持 ROS 2 Humble、Cerebra 浏览器界面、Blockly 和 Python。
- 机械部分全部可打印（推荐 PETG），最小打印体积约 220×220×250 mm；STL 与 BOM 在官网免费下载。
- 软件 AGPL-3.0；文章称已用于超过 150 所学校，社区超过 1800 人。

## 工程实践

- 按官网 STL/BOM 打印采购，装好 Pi 5 后用 Cerebra 或 Blockly 做第一批动作，再接 ROS 2。
- 以上规格与许可证均出自文章叙述，**未逐项核对官方仓库**；动手前请以官方 README/BOM 为准。

## 局限与风险

- 本页仅基于一篇第三方公众号文章，规格数字可能随版本变化。
- 仓库迁移、分叉和许可证以官方当前状态为准。

## 关联页面

- [humanoid-robot](./humanoid-robot.md)
- [open-source-humanoid-hardware](./open-source-humanoid-hardware.md)
- [ros2-basics](./ros2-basics.md)
- [机器人开源项目 18 例（微信 GitHubStore）索引](../overview/robot-open-source-wechat-githubstore-18-projects.md)

## 推荐继续阅读

- [pib-rocks](https://github.com/pib-rocks)

## 参考来源

- [wechat_githubstore_18_open_source_robot_projects.md](../../sources/blogs/wechat_githubstore_18_open_source_robot_projects.md)
