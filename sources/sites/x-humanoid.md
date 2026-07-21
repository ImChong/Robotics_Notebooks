# 北京人形机器人创新中心（X-Humanoid）官网

> 来源归档

- **标题：** 北京人形机器人创新中心有限公司 — 官网
- **类型：** site
- **来源：** 北京人形机器人创新中心（X-Humanoid / Beijing Innovation Center of Humanoid Robotics）
- **链接：** https://x-humanoid.com/
- **开源资料页：** https://x-humanoid.com/opensource.html
- **开源社区（外链）：** https://opensource.x-humanoid-cloud.com/
- **代码组织：** https://github.com/Open-X-Humanoid
- **联系邮箱（组织页）：** opensource@x-humanoid.com
- **入库日期：** 2026-07-21
- **一句话说明：** 中国首个聚焦人形机器人核心技术、产品研发与应用生态的创新中心官网；产品叙事分 **本体（天工 / 天轶）**、**智能平台（慧思开物 / WoW / Pelican）**、**数据服务（RoboMIND 等）** 三层，并链到开源社区与 GitHub。
- **开源状态：** **已开源（多入口）** — 本体 URDF/STEP/SDK/ROS、训练工具链、RL 运控、VLA/VLM、数据集与可视化 SDK 等分散在官网开源页、云端社区与 `Open-X-Humanoid` 组织；以各仓 LICENSE / README 为准。
- **沉淀到 wiki：** 是 → [`wiki/entities/x-humanoid.md`](../../wiki/entities/x-humanoid.md)

---

## 为什么值得保留

- **机构总入口**：官网把「通用机器人平台 / 通用具身智能平台 / 通用数据服务平台」三条产品线写清，是理解后续开源仓矩阵与论文团队署名（X-Humanoid）的主页。
- **开源页清单完整**：`opensource.html` 把天工 Lite/Pro 的 URDF、结构图纸、手册、SDK、ROS 软件系统与 **RoboMIND** 数据集介绍并列，是硬件二次开发的官方清单。
- **与社区/GitHub 分工**：官网偏品牌与产品；[`opensource.x-humanoid-cloud.com`](https://opensource.x-humanoid-cloud.com/) 偏文档/问答/课程；[`Open-X-Humanoid`](https://github.com/Open-X-Humanoid) 偏可克隆代码。

## 官网首页摘录（2026-07-21，Jina Reader）

### 产品三层

| 板块 | 官网表述要点 | 详情入口 |
|------|--------------|----------|
| **通用机器人平台** | 「具身天工 2.0」双电池快换、工业级上肢负载、全自由度下肢；「天轶」轮臂双臂协作，导览/巡检/问答 | [detail/jstg.html](https://x-humanoid.com/detail/jstg.html) |
| **通用具身智能平台** | 「一脑多能 / 一脑多机」；单软件系统兼容机械臂、轮式、人等多构型 | [detail/hskw.html](https://x-humanoid.com/detail/hskw.html)（慧思开物） |
| **通用数据服务平台** | 「场景采集–模型迭代–场景验证」数据飞轮；**RoboMIND**、铰接物品数字资产 **ArtVIP** | [jszndmx.html](https://x-humanoid.com/jszndmx.html) |

### 导航产品树（opensource / 顶栏）

| 类 | 产品 | 链接 |
|----|------|------|
| Embodied | Embodied Tien Kung 3.0 | https://x-humanoid.com/detail/jstg.html |
| Embodied | Tian Yi 2.0（天轶） | https://x-humanoid.com/detail/tianyi.html |
| Intelligent | HuiSi KaiWu（慧思开物） | https://x-humanoid.com/detail/hskw.html |
| Intelligent | Embodied World Model（WoW） | https://x-humanoid.com/detail/world.html |
| Intelligent | Pelican VLM | https://x-humanoid.com/detail/vlm.html |
| Data | Embodied AI Data | https://x-humanoid.com/jszndmx.html |

### 应用实践（首页列举）

工业零部件自主分拣、电控柜质检操作、物品自主搬运、展厅导览等（详情见 [yncj.html](https://x-humanoid.com/yncj.html)）。

## 开源页摘录（`opensource.html`，2026-07-21）

标题区：**天工本体开源 / 数据集开源**。

| 资产 | 说明（官网原文归纳） |
|------|----------------------|
| 天工 Lite / Pro **URDF** | 完整 URDF + STL；关节限位与质量分布；ROS / Gazebo |
| 天工 Lite / Pro **结构图纸** | STEP；部件结构、连接与装配 |
| 天工 Lite / Pro **说明文档** | 开箱—使用—维护手册 |
| 天工 Lite / Pro **SDK** | Lite：电机控制 + IMU；Pro：电机、六维力、IMU、相机接口与调用流程 |
| 天工 **软件系统** | 基于 ROS：`body_control`、`robot_description`、`usb_sbus` 等底层控制 |
| **RoboMIND** | 约 **10.7 万** 真实演示轨迹；**479** 任务；**96** 类物体（官网开源页介绍） |

> **抓取说明：** 直接 `curl` 官网根路径可能触发反爬脚本；入库内容以 Jina Reader 可读正文 + `opensource.html` WebFetch 为准。下载 ZIP 链接在开源页列出，复现时以页面当前链接为准。

## 与本仓库现有资料的关系

- 硬件实体页：[天工 Lite / Pro](../../wiki/entities/tienkung-humanoid-open-source.md)（微信策展第 01 期入口）。
- 方法 / 论文：[Pelican-Unified](../../wiki/methods/pelican-unified-1.md)、[Heracles](../../wiki/entities/paper-heracles-humanoid-diffusion.md)、[HEX（161#038）](../../wiki/entities/paper-loco-manip-161-038-hex.md)。
- 机构标签：`schema/institutions.json` 已注册 `x-humanoid` →「北京人形机器人创新中心（X-Humanoid）」。

## 对 wiki 的映射

- 升格 [`wiki/entities/x-humanoid.md`](../../wiki/entities/x-humanoid.md)：三入口导航 + 开源栈矩阵 + 与 OpenLoong / 商业平台对照。
- 交叉更新 [`wiki/entities/tienkung-humanoid-open-source.md`](../../wiki/entities/tienkung-humanoid-open-source.md)、[`wiki/entities/open-source-humanoid-hardware.md`](../../wiki/entities/open-source-humanoid-hardware.md)。
