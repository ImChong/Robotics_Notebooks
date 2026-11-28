# StackForce 工作台（workbench.stackforce.cc）

> 来源归档

- **标题：** StackForce 机器人强化学习训练向导
- **类型：** site（Web 应用 / 流程向导）
- **来源：** StackForce 轻量级机器人开发平台（stackforce.cc）
- **链接：** https://workbench.stackforce.cc/
- **入库日期：** 2026-07-08
- **全文复核日期：** 2026-11-28
- **核对范围：** 抓取并完整通读 <https://workbench.stackforce.cc/> 当前渲染页面全部文本（94 条交互文案节点）、`https://cad2urdf.stackforce.cc/upload` 与开源硬件配套仓 `Seeed-Projects/Stackforce_Mini_Wheeled_Legged_Robot` `README.md`；核实平台属于闭源在线 SaaS 工具工作台（源码未公开，仅硬件与轮足课程仓库开源；`stackforce.cc` 根页仅返回 SPA Shell）。
- **通读：** 已完成
- **一句话说明：** 按开发者当前资产状态（无模型 / 有 STP / 已有 URDF·USD）分岔引导的 **CAD→URDF→MJLab/Isaac SimReady 在线训练→MuJoCo Sim2Sim 重放→Sim2Real 部署** 在线向导（更新版页面明确支持 **在线 MJLab 训练** 与 **MuJoCo 浏览器版 Sim2Sim 重放**）。
- **沉淀到 wiki：** [stackforce](../../wiki/entities/stackforce.md)

---

## 站点要点（2026-07-08 抓取）

### 定位

- **中文标题：** StackForce 工作台 — 机器人强化学习训练向导
- **核心叙事：** 不是把所有工具堆在一起，而是让读者知道机器人从 **资产、仿真到训练工程** 要经过哪几步。
- **入口分岔（三步状态选择）：**
  1. **没有机器人 3D 模型** → 进入 **URDF 网页版建模**（无需安装）
  2. **有机器人 3D 模型（\*.stp）** → 进入 **CAD2URDF** 转换
  3. **有机器人 3D 模型（\*.urdf / \*.usd）** → **直接输入** 转化为 Isaac 强化学习工程

### 五步主流程（页面文案）

| 步骤 | 动作 | 对应工具/入口 |
|------|------|----------------|
| 第一步 | 建模 / 转换 / 直接输入 | StackForce URDF 网页版；[CAD2URDF](https://cad2urdf.stackforce.cc/upload)；已有 URDF/USD 直传 |
| 第二步 | 导出强化学习代码工程 | **StackForce SimReady** 一键导出 Isaac 强化学习工程 |
| 第三步 | 配置训练环境并开始训练 | 本地环境脚本 / conda；**云训练 GPUFree 镜像** |
| 第四步 | 训练结果重放和评估 | 在 Isaac 中重放运控策略 |
| 第五步 | 实物部署 | 将训练策略部署到机器人硬件 |

### 学习入口（四阶段概念）

1. **机器人资产** — CAD、URDF、mesh 各自职责：几何外形、连杆关节结构、视觉与碰撞资源
2. **可仿真模型** — 补齐到 SimReady 可接收状态：关节、碰撞、惯量、质心、坐标轴可检查
3. **仿真任务** — 在 SimReady 中配置关节、场景、目标动作与奖励思路
4. **训练闭环** — 从仿真反馈迭代模型与参数，形成可复用训练资产

### 内嵌工具

- 页面内 **iframe 嵌入** 线上工具，支持缩放（60%–125%）与 **新窗口打开**
- 与独立站点 [stackforce-cad2urdf.md](stackforce-cad2urdf.md) 及主站 [stackforce.cc](https://stackforce.cc/) 同属 StackForce 生态

---

## 与平台硬件背景

- StackForce 主站定位为 **模块化堆叠式机器人开发平台**（主控 + 电机/舵机/传感器模块），支持 Arduino IDE、PlatformIO、Simulink、Python 与 ROS 接入；Seeed Studio 有基于该平台的 [小轮足开源整机教程](https://wiki.seeedstudio.com/cn/stackforce_mini_wheeled_legged_robot/) 与 [GitHub 仓](https://github.com/Seeed-Projects/Stackforce_Mini_Wheeled_Legged_Robot)。
- 工作台面向 **从自定义整机到 Isaac RL** 的软件链路，与硬件模块销售/教程形成 **「硬件套件 + 云工具链」** 组合。

---

## 2026-11-28 线上页面更新核对

- 原四步向导已显式更新为：**第一步（先建模/转换/直接输入）→ 第二步（在线 MJLab 强化学习训练：SimReady 配置任务、开训与看结果）→ 第三步（Sim2Sim 训练结果重放与评估：StackForce MuJoCo 浏览器版检查迁移）→ 第四步（StackForce Sim2Real 实物部署）**。
- 工具入口固定为三大在线子系统：`CAD2URDF`（STEP/STP 导入与转换）、`机器人 3D 建模`（从零/模板建模、修正 URDF）、`SimReady`（接收 URDF 资产进入仿真）。
- **开源边界核对：** 工作台后端、MJLab 训练服务与 Web 建模/转换源码均不公开；仅其轮足硬件示例与课程 PlatformIO 代码开源于 `Seeed-Projects/Stackforce_Mini_Wheeled_Legged_Robot`。

## 对 wiki 的映射

- 升格页面：[wiki/entities/stackforce.md](../../wiki/entities/stackforce.md)
- 交叉引用：[wiki/entities/step2urdf.md](../../wiki/entities/step2urdf.md)、[wiki/entities/urdf-studio.md](../../wiki/entities/urdf-studio.md)、[wiki/entities/isaac-lab.md](../../wiki/entities/isaac-lab.md)、[wiki/entities/gpufree.md](../../wiki/entities/gpufree.md)、[wiki/concepts/urdf-robot-description.md](../../wiki/concepts/urdf-robot-description.md)

## 参考链接

- 工作台：<https://workbench.stackforce.cc/>
- CAD2URDF：<https://cad2urdf.stackforce.cc/upload>
- 主站：<https://stackforce.cc/>
- Seeed 小轮足 Wiki：<https://wiki.seeedstudio.com/cn/stackforce_mini_wheeled_legged_robot/>
