# wechat_githubstore_18_open_source_robot_projects

> 来源归档（blog / 微信公众号）

- **标题：** 想进入机器人领域吗？学习这18个开源项目就够了
- **类型：** blog
- **来源：** 微信公众号「GitHubStore」（作者署名：小G）
- **原始链接：** https://mp.weixin.qq.com/s/DVGSnatQ5hw2UxAkPeS1Wg
- **发布日期：** 2026-10-27
- **入库日期：** 2026-10-27
- **抓取方式：** `scripts/fetch_wechat_article.py`（Playwright + 预装 Chromium + MicroMessenger UA），纯文本，无图片
- **一句话说明：** 按 1–18 编号列出 18 个开源机器人项目（人形、四足、机械臂、移动平台），每项给出规格、许可证与 GitHub/GitLab 入口。

## 文中列出的 18 条（编号与原文一致）

| # | 项目 | 类型 | 主入口 | 本库节点 |
|---|------|------|--------|----------|
| 1 | Microban | 小型人形 | [Rhoban/microban](https://github.com/Rhoban/microban) · [mjlab_microban](https://github.com/Rhoban/mjlab_microban) | `microban-humanoid`（新） |
| 2 | MiniLoong（强化小子 / RL-Boy） | 小型人形 | [loongOpen/MiniLoong](https://github.com/loongOpen/MiniLoong) | `miniloong-humanoid`（新） |
| 3 | Poppy Humanoid | 教育人形 | [poppy-project/poppy-humanoid](https://github.com/poppy-project/poppy-humanoid) | `poppy-project-robots`（既有） |
| 4 | Otto DIY | 教育双足 | [OttoDIY](https://github.com/OttoDIY) | `otto-diy-biped`（新） |
| 5 | pib | 3D 打印人形 | [pib-rocks](https://github.com/pib-rocks) | `pib-printable-intelligent-bot`（新） |
| 6 | Mini Pupper | 小型四足 | [mangdangroboticsclub/mini_pupper](https://github.com/mangdangroboticsclub/mini_pupper) | `mini-pupper`（新） |
| 7 | Petoi Bittle | 商业机器人狗 + 开源 OpenCat | [OpenCat-Quadruped-Robot](https://github.com/PetoiCamp/OpenCat-Quadruped-Robot) | `petoi-bittle-opencat`（新） |
| 8 | Stanford Doggo | 准直驱四足 | [StanfordDoggoProject](https://github.com/Nate711/StanfordDoggoProject) | `stanford-doggo-and-pupper`（既有） |
| 9 | SpotMicro | 社区四足 | [GitLab spotmicroai](https://gitlab.com/custom_robots/spotmicroai) | `spotmicro`（新） |
| 10 | OpenQuadruped | 12 DoF 四足 | [adham-elarabawy/open-quadruped](https://github.com/adham-elarabawy/open-quadruped) | `open-quadruped`（新） |
| 11 | Puguette | 高扭矩四足 | [haruto89610/puguette](https://github.com/haruto89610/puguette) | `puguette-quadruped`（新） |
| 12 | SO-ARM100 / SO-101 | LeRobot 机械臂 | [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) | `painode-170-soarm100`（既有） |
| 13 | RoArm-M2 | 桌面机械臂 | [waveshareteam/roarm_m2](https://github.com/waveshareteam/roarm_m2) | `waveshare-roarm-m2`（新） |
| 14 | OpenMANIPULATOR-X | 教育/科研机械臂 | [ROBOTIS-GIT/open_manipulator](https://github.com/ROBOTIS-GIT/open_manipulator) | `robotis-open-manipulator-line`（既有） |
| 15 | LeKiwi | 移动操作 | [SIGRobotics-UIUC/LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi) | `lekiwi-mobile-manipulator`（新） |
| 16 | TurtleBot3 Burger | 差速移动机器人 | [ROBOTIS-GIT/turtlebot3](https://github.com/ROBOTIS-GIT/turtlebot3) | `turtlebot3`（既有） |
| 17 | OpenBot | 手机驱动小车 | [ob-f/OpenBot](https://github.com/ob-f/OpenBot) | `paper-openbot-smartphone-mobile-robot`（既有） |
| 18 | Duckiebot | 自动驾驶教学小车 | [duckietown](https://github.com/duckietown) | `duckietown-duckiebot`（新） |

## 各项关键事实（原文摘要）

- **1 Microban：** 约 30 cm；19 个 XL330-M288-T 舵机 + Pi Zero 2W；PLA 全打印；目标成本 550–600 美元；Onshape 装配模型；Makefile+SSH 控制；mjlab_microban 速度跟踪策略可上真机。
- **2 MiniLoong：** 约 55 cm、22 DoF；全链路开源含 ONNX 部署框架；Isaac 同步；Apache-2.0。
- **3 Poppy：** 约 25 DoF、Dynamixel、可动脊椎；硬件 CC BY-SA、软件 GPL v3。
- **4 Otto：** Arduino 兼容；基本版 4 舵机；OttoDIYLib、Blockly；MIT/GPL。
- **5 pib：** pib#5、Pi 5、ROS 2 Humble；PETG；AGPL-3.0；>150 所学校。
- **6 Mini Pupper：** 3-DoF coreless 舵机；ROS/ROS2、OAK-D、MuJoCo/Isaac；Apache-2.0。
- **7 Bittle：** BiBoard+ESP32；35+ 动作；**硬件非开源 DIY 设计**。
- **8 Doggo：** <5 kg；Teensy 3.5 + ODrive；官方称成本 <3000 美元。
- **9 SpotMicro：** 12 舵机；主仓库迁 GitLab。
- **10 OpenQuadruped：** Teensy；Bezier 步态；MIT。
- **11 Puguette：** 12 DoF、单关节约 55 Nm；MIT。
- **12 SO-ARM100：** Feetech STS3215；载荷约 500 g、臂展约 300 mm；Apache-2.0。
- **13 RoArm-M2：** <900 g；0.5 kg@0.5 m；硬件为商业套件。
- **14 OpenMANIPULATOR-X：** 5 DoF、XM430-W350-T；载荷 500 g；重复精度 <0.2 mm；Apache-2.0。
- **15 LeKiwi：** Kiwi 底盘 + SO-ARM；Apache-2.0。
- **16 TurtleBot3 Burger：** LDS-02 LiDAR；OpenCR 固件。
- **17 OpenBot：** 车体约 50 美元；MIT。
- **18 Duckiebot：** gym-duckietown；ROS。

## 对 wiki 的映射

- 索引页：`wiki/overview/robot-open-source-wechat-githubstore-18-projects.md`；每个项目**只对应一个**实体节点（上表「本库节点」列），既有节点只追加本来源与互链，不新建重复页。

## 可信度与使用边界

- 本文为第三方清单，**未逐项核对官方仓库**（规格、星标、许可证、仓库迁移以官方 README 为准）。
- 第 7 项 Bittle 文中明确：硬件非开源 DIY 设计；第 13 项 RoArm-M2 硬件为商业套件。
