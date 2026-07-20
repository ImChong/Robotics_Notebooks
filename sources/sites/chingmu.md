# 青瞳视觉 CHINGMU（en.chingmu.com）

> 来源归档（ingest）

- **标题：** 上海青瞳视觉科技有限公司官网（CHINGMU Vision）
- **类型：** site
- **链接：** https://en.chingmu.com/（中文：<https://www.chingmu.com/>）
- **机构：** 上海青瞳视觉科技有限公司（SHANGHAI CHINGMU VISION TECHNOLOGY CO., LTD）
- **入库日期：** 2026-07-20
- **一句话说明：** 国内自研 **光学动捕全栈**（算法 + 硬件 + 软件）供应商；在 **具身智能 / 人形机器人** 场景提供性能评测、灵巧手、遥操作与 **DAQ 数据采集场** 方案，并运营 **MotionDecode** 千小时人体运动数据开放计划。

## 公司概况（据官网 About / 首页）

- **成立：** 2015 年；宣称 **核心算法—硬件—软件 100% 自研**，研发人员占比约 **70%**。
- **定位：** 光学动作捕捉与 3D 智能感知；业务覆盖 **具身智能、工程科研、虚拟现实、生命科学、数字娱乐、青瞳教育** 等。
- **客户叙事：** 已为 **1000+** 高校与企业机构提供方案（官网列举华为、腾讯、清华等）。
- **国家项目：** 参与 **10+** 项国家重点研发计划（官网表述）。

## 硬件产品线（Hardware）

| 系列 | 角色（官网导航） |
|------|------------------|
| **K Series** | 旗舰光学动捕相机（如 K26；百人大场景案例） |
| **MC Series** | 标准/灵活配置光学相机 |
| **R Series** | 参考相机（RGB、时间码对齐；部分型号支持 markerless / marker 双模） |
| **D / U Series** | 其他相机形态 |
| **Prometheus** | 独立产品线 |
| **PULSEH** | 光惯融合手套（遥操作页提及，亚毫米级手指追踪） |

## 软件栈（Software）

| 产品 | 用途（官网归纳） |
|------|------------------|
| **CMAvatar**（Ulti / AITheta / Bio） | 实时 6DoF 跟踪、可视化、同步与定量分析 |
| **CMTracker / CMVision / CMProcess / CMCapture** | 跟踪、视觉、处理与采集子模块 |
| **SDK** | 二次开发与系统集成 |

## 机器人应用（Applications → Robotics）

官网将机器人场景拆为四条主线：

1. **Performance evaluation** — 以动捕为真值，量化人形/仿生/外骨骼 **轨迹精度、重复性、姿态稳定性、多机协同误差**；对接 ROS/ROS2、MATLAB/Simulink、Python 与常见仿真工具。
2. **Dexterous Hand** — 人/机双手 6DoF 实时跟踪，服务 **模仿学习、遥操作、HRI、精细操作** 的数据采集与评测。
3. **Teleoperation** — 亚毫米人体映射 + 毫秒级闭环，集成 **PULSEH 手套** 与 CMAvatar 统一枢纽。
4. **Data Acquisition Facility（DAQ）** — 面向具身 AI 的 **多模态数据采集场**：运动、视觉、控制信号、触觉、物体状态与环境信息关联；提供从选型部署到标注质检的 **全链路服务**。

## MotionDecode 数据开放计划

- **新闻入口：** [Free Application! 1000-Hour Dataset](https://en.chingmu.com/company-news/10746.shtml)（2026-06-15）
- **规模叙事：** 首期开放申请 **1000 小时** 高质量人体运动数据；总储备约 **3000 小时**。
- **模态：** Body｜Hand｜Ego｜Exo｜Object 6D｜Scene。
- **格式：** BVH、FBX、CSV 等 **10+** 标准格式。
- **任务：** 运动生成、模仿学习、**运动重定向**、仿真验证、技能学习。
- **场景：** 基础移动、工业制造、家庭服务、物流仓储、零售、灵巧操作、文娱等。
- **获取方式：** 样例数据在 **GitHub、Hugging Face 与官网** 滚动发布；完整库需 **申请表**（<https://v.wjx.cn/vm/rqsTPkU.aspx>）或邮件 **MotionDecode@chingmu.com**。
- **公开数据集仓：** [CMRobot/MotionDecode](https://huggingface.co/datasets/CMRobot/MotionDecode)（详见 [`cmrobot-motiondecode.md`](../repos/cmrobot-motiondecode.md)）。

## 工程里程碑（公开案例）

- **CHINGMU × AMD 百人实时动捕纪录**（2026-05-31，[新闻](https://en.chingmu.com/company-news/10728.shtml)）：上海 MCP Boundless Studio，**1000㎡** 场地、**76 台 K26**、**~5300 markers**、**120 fps**、端到端约 **12 ms**；计算平台为 AMD Threadripper PRO 9985WX + Radeon RX 9070 XT。官网称此前国际认证纪录为 19 人、国内公开演示约 41 人。

## 源码与 SDK 开放核查（步骤 2.5）

| 类别 | 开放程度 | 入口 |
|------|----------|------|
| **核心动捕软件（CMAvatar 等）** | **未开源** | 官网 Support → Software download；商业授权 |
| **集成 SDK** | **部分开源** | GitHub 组织 [ChingMuVisionTech](https://github.com/ChingMuVisionTech)：UE / Unity / C++ VRPN / Python / iClone / MotionBuilder 等插件与示例（见 [`chingmu-github-sdks.md`](../repos/chingmu-github-sdks.md)） |
| **MotionDecode 数据集** | **部分开源** | HF `CMRobot/MotionDecode` 公开样例与索引；完整 1000h 需申请或 HF Request access |
| **训练/推理代码** | **未列** | 截至入库日官网与 HF 均未提供策略训练仓库 |

## 对 wiki 的映射

- [青瞳视觉 CHINGMU](../../wiki/entities/chingmu.md) — 光学动捕厂商 + 具身数据管线实体页
- [Motion Retargeting](../../wiki/concepts/motion-retargeting.md) — MotionDecode 与 DAQ 下游常见步骤
- [Teleoperation](../../wiki/tasks/teleoperation.md) — 遥操作与动捕真值采集对照
- [FreeMoCap](../../wiki/entities/freemocap.md) — 开源低成本 MoCap 对照选型
