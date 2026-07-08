# StackForce 工作台（workbench.stackforce.cc）

> 来源归档

- **标题：** StackForce 机器人强化学习训练向导
- **类型：** site（Web 应用 / 流程向导）
- **来源：** StackForce 轻量级机器人开发平台（stackforce.cc）
- **链接：** https://workbench.stackforce.cc/
- **入库日期：** 2026-07-08
- **一句话说明：** 按开发者当前资产状态（无模型 / 有 STP / 已有 URDF·USD）分岔引导的 **CAD→URDF→Isaac 强化学习工程→训练→评估→实物部署** 在线向导；内嵌 StackForce 自有 Web 工具（URDF 建模、CAD2URDF、SimReady 工程导出）并串联本地与云训练入口。
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

## 对 wiki 的映射

- 升格页面：[wiki/entities/stackforce.md](../../wiki/entities/stackforce.md)
- 交叉引用：[wiki/entities/step2urdf.md](../../wiki/entities/step2urdf.md)、[wiki/entities/urdf-studio.md](../../wiki/entities/urdf-studio.md)、[wiki/entities/isaac-lab.md](../../wiki/entities/isaac-lab.md)、[wiki/entities/gpufree.md](../../wiki/entities/gpufree.md)、[wiki/concepts/urdf-robot-description.md](../../wiki/concepts/urdf-robot-description.md)

## 参考链接

- 工作台：<https://workbench.stackforce.cc/>
- CAD2URDF：<https://cad2urdf.stackforce.cc/upload>
- 主站：<https://stackforce.cc/>
- Seeed 小轮足 Wiki：<https://wiki.seeedstudio.com/cn/stackforce_mini_wheeled_legged_robot/>
