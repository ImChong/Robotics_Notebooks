# 国内具身开源全景 76 家 · 424 项 长文通读（具身智能研究室）

> 来源归档（原文通读 · 支撑 `wiki/overview/china-domestic-embodied-opensource-76-companies-technology-map.md`）

- **标题：** 国内具身智能的开源全景，收藏这一篇就够了：76 家公司、420+ 个开源项目
- **作者：** 具身智能研究室（Yuanxq）
- **类型：** blog / survey / open-source-landscape
- **原文：** <https://mp.weixin.qq.com/s/L2XQBhesU8EiS2nKM7HErw>（2026-09-06；全文抓取见 [wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md](../raw/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)）
- **一句话说明：** 按五层格局逐机构、逐仓库列出国内具身开源项目，每条含类别标签与一句话说明；开篇给出各层定位与"开源最成体系 / 最热 / 最不缺"的三句判断。
- **核对范围：** 通读抓取全文 1054 行：开篇与五层格局论述、第一至第五层全部机构的逐仓库条目（含类别标签与说明）、结尾。未逐个打开清单中被引用的 GitHub / Gitee / HF 仓库核对其实际代码与许可；各仓库能力描述仅采信原文一句话说明（其中部分条目为模板化描述，如"官方 SDK：真机控制与状态读取的统一接入层"）。机构数、项目数以原文标题与各机构括号计数为准，未逐条重新计数。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **总判断：** 不到两年，76 家机构公开 420+ 项目，覆盖真机数据集、VLA、世界模型、训练框架、仿真与 SDK；"开源最成体系的是头部整机厂，最热的是模型公司的 VLA 与世界模型，最不缺的是 SDK 驱动"。
- **第一层 整机厂商与创新中心（27 家）：** 开源"整条链路"——本体资产、RL 训练、仿真、部署 SDK 全套公开；"代码是生态入口，护城河在硬件与数据"。典型链路：智元 agibot_x1 的 train / infer / hardware；宇树 Unitree RL Lab / RL Gym / RL Mjlab + unitree_mujoco + unitree_sdk2；北京人形 TienKung-Lab（AMP + Isaac Lab + MuJoCo Sim2Sim + ROS 2 部署）与 xMimic（BVH→PKL→NPZ→PPO 跟踪→ONNX）；傅利叶 Wiki-GRx 四步流水线（模型、Isaac Gym、MuJoCo、真机）；加速进化 Booster Gym；星动纪元 Humanoid-Gym；逐际 TRON1 RL；云深处 Lite3 RL Deploy 等。
- **第二层 模型 / VLA / 世界模型 / 数据（21 家）：** 开源"大脑"与"燃料"，目的是立标准、圈开发者（"LeRobot 数据格式已成事实标准"）。条目含银河通用 GraspVLA / Any2Track、自变量 WALL-X / WALL-WM、极佳视界 GigaWorld / GigaBrain、蚂蚁灵波 LingBot 系列（VLA、世界模型、深度、地图）、原力灵机 OpenDM / OpenDW、求之科技 DISCOVERSE（3DGS + MuJoCo Real2Sim2Real）等。
- **第三层 灵巧手与触觉（6 家）：** 手部 SDK、仿真、遥操作与触觉数据工具链，"开源浓度高"；如灵心巧手 LinkerHand、舞肌科技 Wuji Hand（含 wuji-mjlab 手内旋转 PPO 与 Sim2Real）、帕西尼 DF-1/2/3 触觉数据产线、千觉 Xense 触觉 SDK、傲意肌电臂环 / ROHand。
- **第四层 大厂与底层平台（6 家）：** 战略卡位，多从模型层与工具链切入——阿里（ABot / Rynn 系列）、腾讯 HY-Embodied、字节 GR-1 / GR-MG、百度 LoongForge、地平线 HoloMotion / HoloAgent / RoboOrchard、地瓜 RDK 部署工具。小米 / 小鹏条目被原文归入第一层机构列表。
- **第五层 产业链与移动平台（16 家）：** 机械臂、相机、雷达、控制器厂商把 SDK 与 ROS 驱动当作默认交付物，如松灵、众为 xArm、大象 myCobot、奥比中光、禾赛、速腾、非夕 Flexiv RDK、睿尔曼等；诺亦腾 AdaPT 仓库仅含第一阶段 Tracker，论文其余模块未开放。
- **原文自述的开放边界（可供局限小节引用）：** WALL-SS 目前只是论文与图示，"不能作为可直接训练或部署的代码框架"；Pelican-VLA 0.5 承认"从表征到可靠动作仍有缺口"；HoloAgent 模型与数据分发尚未完成；Mini Pi Plus AMP 的 README 明确"面向研究与仿真，实机连接前需受控验证"；云深处训练环境"不把训练环境误写成完整实机控制栈"；Unitree RL Gym（Isaac Gym 时代）"不应与 Isaac Lab 版本混用"。

## 对 wiki 的映射

- [`wiki/overview/china-domestic-embodied-opensource-76-companies-technology-map.md`](../../wiki/overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [`wiki/queries/china-domestic-opensource-424-coverage.md`](../../wiki/queries/china-domestic-opensource-424-coverage.md)
