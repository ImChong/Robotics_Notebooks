# AimRT

> 来源归档（ingest · 官方 GitHub + 文档）

- **标题：** AimRT — A high-performance runtime framework for modern robotics
- **类型：** repo
- **机构：** 智元机器人（AgiBot）
- **链接：** <https://github.com/AimRT/AimRT>
- **文档：** <https://docs.aimrt.org/> · 门户 <https://aimrt.org/>
- **许可：** Mulan PSL v2（仓库 LICENSE 文件）
- **Stars：** ~1408（2026-09-23 核查）
- **Topics：** cpp20, robotics
- **分类：** 部署运行时
- **入库日期：** 2026-09-06（初归档）；2026-09-23 深度 ingest 补充 README 与架构要点
- **一句话说明：** 面向现代机器人领域的 **Modern C++ 运行时开发框架**：轻量易部署，在资源管控、异步编程、部署配置方面采用现代设计；整合端侧/边缘/云端研发，兼容 ROS 2、HTTP、gRPC，并提供插件化扩展与可观测性工具链。
- **沉淀到 wiki：** [`wiki/entities/cn-os-aimrt.md`](../../wiki/entities/cn-os-aimrt.md)

## 开源状态

- **已开源**：GitHub 公开仓库 <https://github.com/AimRT/AimRT>，Mulan PSL v2。
- **文档站：** <https://docs.aimrt.org/>（快速开始、发布说明、联系方式）。
- **定位边界：** 运行时与通信/部署基础设施 — **不直接提供运动策略或训练代码**；与 [`aimrt_mujoco_sim`](../../wiki/entities/mujoco.md) 等仿真联调入口、以及 X2 机载 [AimDK](../sites/x2-aimdk-agibot.md) 应用层接口分工不同。

## README 核心要点（2026-09-23）

1. **Modern C++ 基础运行时** — 轻量、易部署；资源管理、异步编程、部署配置等现代设计。
2. **跨部署场景整合** — 机器人端侧、边缘端、云端统一研发路径；服务 AI/云原生机器人应用。
3. **可观测性与调试** — 完善的调试、性能分析工具链。
4. **插件化与生态兼容** — 全面插件开发接口；兼容 ROS 2、HTTP、gRPC；支持对现有系统的渐进式升级。

## 智元生态中的位置

| 层级 | 代表 | 关系 |
|------|------|------|
| 运行时 | **AimRT** | 模块通信、线程/资源组织、日志监控 |
| 真机 SDK | [AimDK X2](../sites/x2-aimdk-agibot.md) | X2 二次开发 API（ROS 2 上层） |
| 仿真联调 | aimrt_mujoco_sim | MuJoCo 动力学 + AimRT 通信/runtime |
| 本体资产 | agibot_x2_urdf | URDF/MJCF/USD 供仿真引用 |

## 对 wiki 的映射

- [wiki/entities/cn-os-aimrt.md](../../wiki/entities/cn-os-aimrt.md)
- 交叉：[wiki/entities/agibot-aimdk-x2.md](../../wiki/entities/agibot-aimdk-x2.md)、[wiki/overview/china-domestic-embodied-opensource-76-companies-technology-map.md](../../wiki/overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
