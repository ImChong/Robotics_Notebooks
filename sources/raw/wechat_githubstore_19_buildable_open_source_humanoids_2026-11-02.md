# 19款能自己造的开源人形机器人

> 原始抓取（2026-11-02，`scripts/fetch_wechat_article.py`），仅供溯源；来源：https://mp.weixin.qq.com/s/UUwUAcKdVrlo-N60Oar3Xg

19款能自己造的开源人形机器人，附完整开源地址
Original 小G GitHubStore
 Nov 2, 2026, 12:03 AM
1. RoboPrime（约 $60–100）

由 simonepri 设计的全开源 21 自由度 3D 打印人形机器人。主控使用 ATmega328P（Arduino Pro Micro），搭配 21 个 MG90S 微型舵机、HC-05 蓝牙模块和自制电路板。整机成本极低，适合初学者从零开始打印、焊接和编程。项目包含完整硬件清单、接线说明和固件，支持基本动作与蓝牙控制，体积紧凑，是入门级桌面人形的典型代表。

开源地址：https://github.com/simonepri/roboprime

2. Qmini（价格未确认）

Unitree 官方开源的低成本双足机器人。整机机械结构可完全 3D 打印，使用 11 个 Unitree 8010 电机（10 个用于腿部行走，1 个颈部用于扩展）。设计强调模块化，类似乐高式组装，熟练者约 3–5 小时可完成。提供完整 BOM、电气框图、STEP 文件、装配 SOP 和 URDF 模型。软件方面有配套的 RoboTamer 相关仓库支持控制与强化学习策略部署。适合想快速入手高质量双足平台的爱好者和教育场景。

开源地址：https://github.com/unitreerobotics/Qmini  
相关软件：https://github.com/vsislab/RoboTamerSdk4Qmini

3. Zowi（约 €100）

西班牙 BQ 公司推出的小型教育双足机器人，完全开源且强调“可深度改装”。使用 FreeCAD 设计，机械件可 3D 打印。支持跳舞、距离检测、声音交互、LED 表情等功能，主控可重新编程。社区有丰富的固件库和二次开发案例，非常适合中小学教学和爱好者入门。项目历史较长，资源相对成熟。

开源地址：https://github.com/JavierIH/zowi  
官方库：https://github.com/bq/zowiLibs

4. Otto Humanoid / Otto DIY（约 $50 起）

最经典的开源教育机器人之一。基于 Arduino Nano 和 4 个 SG90 舵机，整机 3D 打印，成本极低。支持行走、跳舞、避障、声音互动等多种行为。官方提供完整 Arduino 库（OttoDIYLib），社区有大量 ESP32 等升级版本和教程。非常适合编程初学者、STEM 教育和个人快速原型。

开源地址：https://github.com/OttoDIY/OttoDIYLib  
主项目：https://github.com/Open-Hardware-Leaders/OttoDIY

5. Kayra（价格未确认）

由 Ramin Assadollahi 发起的社区驱动 3D 打印开源人形。使用 FreeCAD 设计，目标是低成本、易修改、可步行，并支持多种控制器（Arduino、ESP32 等）。项目包含设计文件、打印指南、接线图和仿真环境，强调“社区共同进化”。适合想参与设计迭代、自定义结构的爱好者。

开源地址：https://github.com/assadollahi/kayra 

6. PLEN2（套件约 $499）

日本 PLEN Project 推出的可打印开源人形机器人，号称较早的完整可打印开源人形方案之一。提供完整 STL 文件、固件和动作库，支持基本行走和动作回放。体积紧凑，适合学习和二次开发。社区也有使用不同舵机（如 MG90S）的改装版本。

开源地址：https://github.com/plenprojectcompany/PLEN2

7. Nova Desktop Humanoid（约 $150）

由 Hayden Woodger（United Robotics）开发的桌面陪伴型人形。3D 打印结构，使用少量 ST3215 等舵机实现头部、手臂和躯干的表达性动作。支持人脸追踪、语音对话、表情动画和记忆功能，强调“像有存在感的桌面伙伴”。设计文件可在 Printables、Creality Cloud 等平台获取，适合想做 AI 交互桌面机器人的用户。

地址：官网 https://www.novarobot.co.uk/  
STL 模型可在 Printables / Creality Cloud 搜索 “Nova Desktop” by Hayden Woodger

8. Bambot（约 $300）

低成本开源 AI 机器人项目，强调“让玩机器人变得简单”。提供 3D 打印件、硬件清单和浏览器端控制界面。设计结合常见舵机与模块化结构，Apache-2.0 许可。适合想快速搭建可交互机器人、并尝试 AI 控制的爱好者。

开源地址：https://github.com/timqian/bambot

9. Rapiro（约 $390）

由日本设计师 Shota Ishiwatari 设计的小型人形机器人套件。配备 12 个舵机、Arduino 兼容主板，支持安装 Raspberry Pi 进行扩展（视觉、网络等）。3D 模型公开，社区有多种驱动代码和改装案例。适合想做可编程、可扩展桌面人形的用户。

开源地址：官网 https://www.rapiro.com/  
STL 在 Thingiverse 可搜索 Rapiro；社区驱动示例：https://github.com/hybridgroup/cylon-rapiro

10. Valera（约 $400）

开源社交机器人项目，目标是打造低成本、易构建的 AI 交互平台。提供完整 3D 源文件和 STL，软件与硬件均开放（MIT 许可）。强调社交互动和可及性，适合想做对话型或展示型桌面机器人的用户。

开源地址：https://github.com/goodlancer-org/valera

11. Reachy Mini Lite（约 $399）

Pollen Robotics 与 Hugging Face 合作的开源桌面机器人。紧凑尺寸，支持 Python 编程、LLM 对话、表情动作和社区应用。Lite 版连接外部电脑使用，硬件设计与软件 SDK 均开放。适合 AI 开发者、教育场景和想快速接入开源模型生态的用户。

开源地址：https://github.com/pollen-robotics/reachy_mini  
官网：https://pollen-robotics.com/reachy-mini/

12. Open Duck Mini（低于 $400）

开源迷你双足机器人（BDX 风格），约 42 cm 高。完全 3D 打印，提供强化学习行走策略（ONNX）、运行时和设计文件。项目强调 sim-to-real，适合对强化学习、双足行走感兴趣的爱好者和开发者。

开源地址：https://github.com/apirrone/Open_Duck_Mini  
Runtime：https://github.com/apirrone/Open_Duck_Mini_Runtime

13. MicroDuck（约 $399）

Pollen Robotics 的迷你双足鸭形机器人，约 25 cm 高、约 800 g。基于强化学习策略（MuJoCo + PPO）驱动，支持行走、滚动、抓取等动作。完整开源，包含运行时和训练环境，是当前较活跃的小型 RL 人形项目之一。

开源地址：https://github.com/pollen-robotics/microduck  
RL 训练：https://github.com/pollen-robotics/microduck_rl

14. ROBOTIS MINI（约 $540）

基于 DYNAMIXEL XL-320 的紧凑人形机器人套件。官方提供设计文件（STL/STP）、编程工具和详细文档，支持多种控制方式。硬件质量较高，社区有 ROS 等二次开发资源。适合想用成熟舵机生态做实验的用户。

开源地址：官方文档与文件见 https://github.com/ROBOTIS-GIT/emanual（mini 页面）  
社区示例：https://github.com/japonophile/darwin

15. Halley（约 $500–800）

约 2.6 英尺高的开源人形机器人，设计用于人机交互与表情表达（可用手机作为脸部）。提供完整 3D 设计、软件和详细构建教程（GPL）。适合对社交互动、表情机器人感兴趣的用户。

开源地址：https://github.com/johnchoi313/Halley-Ambassador-Robot-001  
详细教程见 Instructables

16. Microban（约 $550–600）

Rhoban 团队（波尔多大学 LaBRI）开发的约 30 cm 全 3D 打印开源人形。使用 19 个 Dynamixel XL330 舵机和 Raspberry Pi Zero 2W，提供完整文档、CAD、装配指南和软件。强调可及性和可修改性，适合学生和爱好者进行学习和实验。

开源地址：https://github.com/Rhoban/microban

17. Endeavor II（约 $600）

3D 打印人形机器人项目，提供组装好的 STEP 模型。持续迭代固件（ESP32 等）和机械结构，适合作为通用人形平台进行二次开发和实验。项目仍在更新中。

开源地址：https://github.com/ProtoDane/EndeavorHumanoid

18. XLeRobot（约 $660）

实用型双臂移动家庭机器人，基于 LeRobot 生态（SO-101 机械臂 + 移动底盘等）。成本约 $660，组装时间短，完整开源。适合家用操作、遥操作和 AI 实验，是当前较受关注的低成本双臂移动平台之一。

开源地址：https://github.com/Vector-Wangel/XLeRobot

19. InMoov（上半身约 $900）

由法国雕塑家 Gaël Langevin 于 2012 年发起的经典全 3D 打印开源人形机器人。全球广泛复刻，支持手部、头部、上半身等模块，使用 MyRobotLab 框架控制。CC-BY-NC 许可，社区资源极其丰富，是开源人形领域的标志性项目。

开源地址：官网 https://inmoov.fr/  
软件框架及相关社区仓库可在 GitHub 搜索 InMoov / MyRobotLab



机器人 · Table of Contents
Previous
想进入机器人领域吗？学习这18个开源项目就够了
​
GitHubStore
Like
Share
Boost
Comment