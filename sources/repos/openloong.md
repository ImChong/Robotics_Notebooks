# OpenLoong（青龙全栈开源项目）

> 来源归档

- **标题：** OpenLoong — 全尺寸人形机器人全栈开源
- **类型：** repo（项目组织 + 多仓生态）
- **来源：** 人形机器人（上海）有限公司、国家地方共建人形机器人创新中心、开放原子开源基金会
- **主入口：** https://github.com/loongOpen/OpenLoong
- **AtomGit 组织：** https://atomgit.com/openloong
- **社区：** https://www.openloong.org.cn/cn/projects/openloong
- **入库日期：** 2026-05-23
- **一句话说明：** 面向「青龙」公版机的四层全栈开源（云端大脑 / 具身小脑 / 具身实体 / 具身数据）；软件以 **ROS-free 分层 C++ 控制框架** 与 **MPC+WBC 动力学控制包** 为主干，并辅以 Isaac Gym RL、ROS1/Gazebo 等历史/并行栈。
- **沉淀到 wiki：** 是 → [`wiki/entities/openloong.md`](../../wiki/entities/openloong.md)

---

## 为什么值得保留

- **唯一公开强调「全尺寸 + 全栈」的开源人形叙事之一**：硬件图纸、动力学控制、仿真部署、数据集与社区运营在同一品牌下维护。
- **双软件主线清晰**：`OpenLoong-Framework`（实机/全链仿真 C++ 栈）与 `OpenLoong-Dyn-Control`（MuJoCo + MPC/WBC 研究栈）分工明确，便于研究者选型。
- **许可与运营主体明确**：主仓声明 **Apache-2.0**；运营方与开放原子项目页可交叉验证规格与定位。

## 四层技术架构（主 README 摘录）

自下而上：

1. **云端大脑（OS）**：多内核（Linux / LiteOS）+ KAL 抽象；HDF 驱动框架。
2. **具身小脑**：具身智能子系统（遥操作、模仿学习、强化学习）+ 全身动力学子系统（仿真、WBC/MPC、数据记录、中间件驱动）。
3. **具身实体**：公版硬件图纸、BOM、装配图；标准化总线协议（README 称 **EthanCat** 分布式总线栈，与硬件 README 的 **EtherCAT** 关节控制需按文档区分「协议品牌 vs 物理层」）。
4. **具身数据**：训推数据集、预训练模型与标准化数据格式（Hugging Face 等逐步发布）。

## 核心组织方式

- 逻辑层级：**系统集 → 子系统 → 软件包**；可按场景裁剪非必要组件。
- 软件包：可复用源码/配置/资源/编译脚本，或可直接运行的二进制（如 `loong_deployment` 预编译部署包）。

## 核心仓库矩阵（GitHub `loongOpen` 组织，2026-05）

### A. 控制框架（实机 / 全链仿真，ROS-free）

| 仓库 | 角色 |
|------|------|
| [OpenLoong-Framework](https://github.com/loongOpen/OpenLoong-Framework) | 子组件索引与编译说明（**独立打开各子仓**） |
| [loong_utility](https://github.com/loongOpen/loong_utility) | C++ 基础库（矩阵/日志/UDP 等） |
| [loong_third_party](https://github.com/loongOpen/loong_third_party) | LGPL 第三方（如 ethercat、modbus）独立编译 |
| [loong_driver_sdk](https://github.com/loongOpen/loong_driver_sdk) | 硬件驱动 SDK → `libloong_driver_sdk_*.so` |
| [loong_ctrl_locomotion](https://github.com/loongOpen/loong_ctrl_locomotion) | 运动控制 → `libnabo_*.so`（状态机 + 全身关节控制） |
| [loong_base](https://github.com/loongOpen/loong_base) | 主程序：`loong_driver` / `loong_interface` / `loong_locomotion` |
| [loong_sim](https://github.com/loongOpen/loong_sim) | 算法仿真：`loong_share_sim_*` |
| [loong_sim_sdk_release](https://github.com/loongOpen/loong_sim_sdk_release) | 预编译全链仿真 SDK + Python 示例 |
| [loong_deployment](https://github.com/loongOpen/loong_deployment) | 预编译实机部署框架（拷贝即用） |

**框架特点（Framework readme）：** 除 IGH EtherCAT 主站外自集成依赖；免 ROS；仿真—实机接口一致；模块化严封装。

### B. 全身动力学控制（MuJoCo 3 MPC/WBC 研究栈）

| 仓库 | 角色 |
|------|------|
| [OpenLoong-Dyn-Control](https://github.com/loongOpen/OpenLoong-Dyn-Control) | 基于 **MPC + WBC**；内置 MuJoCo/Pinocchio/Eigen 等；demo：行走 / 跳跃 / 盲踩障碍 |
| AtomGit 镜像 | https://atomgit.com/openloong/openloong-dyn-control |

- 推荐环境：Ubuntu 22.04、g++ 11.4
- 实机已验证：**行走**、**盲踩障碍**（README）
- 文档：[官网 API](https://www.openloong.org.cn/pages/api/html/index.html)、[Wiki](https://www.openloong.org.cn/pages/wiki/html/index.html)

### C. 强化学习 / 仿真并行栈

| 仓库 | 角色 |
|------|------|
| [OpenLoong-Gymloong](https://github.com/loongOpen/OpenLoong-Gymloong) | **Isaac Gym** 训练青龙行走（AzureLoong 包；Ubuntu 20.04 + CUDA） |
| [OpenLoong-ROS](https://github.com/loongOpen/OpenLoong-ROS) | ROS1 + Gazebo 历史栈（`azureloong_control` / `azureloong_description`） |
| [Unity-RL-Playground](https://github.com/loongOpen/Unity-RL-Playground) | Unity RL 与 embodied 仿真 playground |

### D. 硬件 / 数据 / 上层能力

| 仓库 | 角色 |
|------|------|
| [OpenLoong-Hardware](https://github.com/loongOpen/OpenLoong-Hardware) | 公版机 PDF 图纸与硬件说明 |
| [OpenLoong-Dataset](https://github.com/loongOpen/OpenLoong-Dataset) | Hugging Face 真机操作 episode（koch_* 系列等） |
| [OpenLoong-Brain](https://github.com/loongOpen/OpenLoong-Brain) | 大模型技能调度 / 多模态对话与机器人 IP 指令下发 |
| [manipulation](https://github.com/loongOpen/manipulation) / [navigation](https://github.com/loongOpen/navigation) | 操作与导航子系统 |
| [MiniLoong](https://github.com/loongOpen/MiniLoong) / [NanoLoong-Bipedal](https://github.com/loongOpen/NanoLoong-Bipedal) | 小型化 / 双足衍生平台 |

### E. 社区与资料

| 仓库 | 角色 |
|------|------|
| [community](https://github.com/loongOpen/community) | 社区文档与贡献指南 |
| [awesome-humanoid-robots](https://github.com/loongOpen/awesome-humanoid-robots) | 人形机器人资料清单 |

## 开放原子项目页补充规格

- 公版机：**约 1.85 m** 全尺寸、**强对标人**、公开资料写 **约 43 DOF**、自重约 **85 kg**。
- 三大开源块：Hardware System / Dynamic Control System / Dataset。
- 能力关键词：多种全身遥操作、仿真—实机一致部署、分层模块化二次开发。

## 对 wiki 的映射

- 主实体 [`wiki/entities/openloong.md`](../../wiki/entities/openloong.md)：Mermaid 全栈图 + 仓库链接表 + 软件/硬件分节。
- 硬件 source：[`sources/repos/openloong_hardware.md`](openloong_hardware.md)
- 社区 source：[`sources/sites/openloong_community.md`](../sites/openloong_community.md)
