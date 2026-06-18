# BotWorld（机器人资产平台）

> 来源归档

- **标题：** BotWorld — Robotics Asset Platform / 机器人资产平台
- **类型：** site（具身智能资产社区 + 多工具聚合入口）
- **URL：** <https://botworld.enkeebot.com/>
- **运营主体：** EnkeeBot（域名 `enkeebot.com` / `enkeebot.cn`；ICP 备案主体为北京源奇智能科技有限公司）
- **入库日期：** 2026-06-18
- **一句话说明：** 一站式机器人资产平台：在广场发现、上传、审核与复用 **机器人模型、动作/数据资产、技能、场景与案例**；内置 **URDF Studio / Motion Studio / BotLab** 工作区入口，并聚合 **step2urdf、Motrix Viewer、BridgeDP Engine** 等生态工具。
- **沉淀到 wiki：** [BotWorld](../../wiki/entities/botworld.md)

---

## 平台定位（策展）

- 官方 meta 描述：**「Discover, share, and reuse robot models, motion data, skills, scenarios, and cases」**；中文口号 **「让机器人资产被发现、复用和分发」**。
- 站点自述：**「An embodied intelligence asset community for curating and sharing robot models and data assets.」**
- 与纯 GitHub 仓库索引不同，BotWorld 强调 **可检索广场 + 资产包下载 + 审核发布 + 收藏点赞 + 导入工作区** 的闭环。

## 资产类型与格式（来自前端 i18n / 广场分类）

### 模型类资产

| 类别 | 说明 |
|------|------|
| **Full Robots**（机器人本体） | 整机或完整机构 |
| **End-effectors**（末端执行器） | 夹爪、灵巧手等 |
| **Sensors**（传感器） | 相机、IMU 等模型 |
| **Actuators**（关节执行器） | 电机/关节模组描述 |
| **Articulated Objects**（环境物体） | 可动物体/场景道具 |

**描述格式：** URDF、MJCF、SDF、XACRO、USD

**机体形态标签（示例）：** Humanoids、Quadruped、Bipedal、Wheel-Leg、Chassis、Manipulator、Dexterous Hand、Gripper、Sensor

**厂商/品牌合集（广场 banner 示例）：** Unitree、Stella-Robot、Damiao、Agibot、DeepRobotics、Booster、HighTorque、CASBot、Boston Dynamics、Fourier、KUKA、FANUC、ABB、Zaowu 等

### 数据类资产（Data Gallery）

| 数据语义 | 代表格式 |
|----------|----------|
| Motion Capture（动捕） | BVH、FBX、CSV 等 |
| Retargeting（映射） | NPZ、PKL、JSON 等 |
| Teleoperation（遥操） | CSV、JSON 等 |
| VLA 数据 | 平台自定义包 |
| RL Trajectory（RL 轨迹） | CSV、NPZ 等 |

数据模态标签示例：Motion、Vision、Semantic、Tactile/Force

技能/能力标签示例：Simulation、Control、Vision、Planning

## 广场与工作区

| 分区 | 作用 |
|------|------|
| **Asset Gallery** | 主广场：浏览、搜索、收藏、点赞、下载资产包 |
| **Data Gallery** | 数据资产专区 |
| **Project Gallery / BotLab Projects** | BotLab 相关项目展示 |
| **URDF Studio** | Web 机器人设计与组装工作站（[OpenLegged/URDF-Studio](https://github.com/OpenLegged/URDF-Studio)） |
| **Motion Studio** | 动作预览、编辑、调试与复用 |
| **BotLab** | 机器人仿真 Runtime：资产加载、轻量仿真、物理验证、调试运行 |
| **OMO（Onboard Magic OS）** | 真机固件烧录、遥测与调试（前端标注 **Under Development**） |

资产详情页常见能力：**README 展示、Asset Bundle 下载、Import to Workspace（导入对应 Studio）**、推荐搭配资产。

## 上传与审核流程

1. 登录后可上传资产（支持 **D-Robotics SSO**：`sso.d-robotics.cc`）。
2. 生命周期：**draft（草稿）→ pending review（待审核）→ published（已发布）**；已发布资产更新可走 **待审核更新** 分支。
3. **BotPilot**：平台内置 AI 审核助手（Requested / Reviewing / Pass / Unsure / Rejected 等状态）；登录用户可减少人工验证摩擦。
4. 配额：单用户 **draft + pending + delisted** 状态资产上限 **5** 个（前端文案）。
5. 资产包需包含 **README**、缩略图与主文件；支持 **Asset Bundle** 打包下载。

## 插件中心（Recommended Plugins）

广场「Plugin Center」聚合生态工具，部分 **built-in** 挂在 URDF Studio，部分 **external** 跳转独立站点：

| 工具 | 作者 | 挂载应用 | 外链 |
|------|------|----------|------|
| AI Inspection / AI Conversation / Collision Optimizer | URDF Studio | urdf-studio | 内置 |
| Motion Tracking | Axellwppr | motion-studio | <https://motion-tracking.axell.top/> |
| STEP2URDF | Democratizing Dexterity | — | <https://step2urdf.top/> |
| RoboGo | D-Robotics | botlab | <https://robogo.d-robotics.cc/> |
| Motrix Viewer | Motphys | botlab | <https://motrix.motphys.com/> |
| Trajectory Editing | cyoahs | motion-studio | <https://motion-editor.cyoahs.dev/> |
| BridgeDP Engine | 桥介数物 | motion-studio | <https://engine.bridgedp.com/> |

## 对 wiki 的映射

- [BotWorld（实体页）](../../wiki/entities/botworld.md) — 新建平台总览
- [URDF-Studio](../../wiki/entities/urdf-studio.md) — BotWorld 内置核心建模工作站
- [BotLab / MotionCanvas](../../wiki/entities/botlab-motioncanvas.md) — BotWorld「BotLab」分区对应产品
- [step2urdf](../../wiki/entities/step2urdf.md) — 插件中心外链工具
- [Motrix](../../wiki/entities/motrix.md) — 插件中心 Motrix Viewer 入口

## 参考链接

- 平台首页：<https://botworld.enkeebot.com/>
- URDF Studio GitHub：<https://github.com/OpenLegged/URDF-Studio>
- D-Robotics 社区：<https://www.d-robotics.cc/>
