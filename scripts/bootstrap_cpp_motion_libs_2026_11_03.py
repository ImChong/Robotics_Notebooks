#!/usr/bin/env python3
"""Bootstrap ingest: Qt历险记《C/C++ 运动控制开源库分类整理》（2026-11-02）。

- 文章是 C/C++ 运动控制开源库清单（文章称 5 类，实际点名 15 项，其中 14 项有仓库 / 官方文档）。每个项目都回到一手资料核对：
  仓库 / 官方文档、LICENSE 文件、最近提交、论文；文章与一手资料的差异在节点里标出。
- 新建 7 个节点（Ruckig、pyplcopen=lusipad/plcopen、FastAccelStepper、Robotics Library、Control Toolbox、
  RTmotion、OMPL），复用 7 个（Pinocchio、Crocoddyl、SOEM、IgH EtherCAT Master、SimpleFOC、ros2_control、MoveIt 2）；
  「WBC Whole-Body Control」文章未给仓库，只链概念页。
- 核对方式（入库日）：git ls-remote + 浅克隆读 LICENSE / README（只读）；论文用 arXiv 摘要页核对标题；
  RTmotion 用 Intel ECI 官方文档（未找到公开仓库）。
- 用法：python3 scripts/bootstrap_cpp_motion_libs_2026_11_03.py <抓取的 txt>
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()
URL = "https://mp.weixin.qq.com/s/RIXXIltQ0ABoBO_OVLZTZw"
BLOG = "wechat_qt_adventure_cpp_motion_control_libs_2026-11-02.md"
MAP = "cpp-motion-control-open-source-libraries-map.md"
MAP_TITLE = "C/C++ 运动控制开源库地图：轨迹生成、动力学、工业总线与 ROS 栈（一手资料核对版）"

CAT_TRAJ = "轨迹规划库"
CAT_DYN = "机器人动力学 / 整机控制库"
CAT_BUS = "工业总线与控制栈"
CAT_ROS = "ROS / ROS2 运动控制栈"

# 复用节点（库内已有）：一手核对结论（追加到节点「一手资料核对」小节）
REUSE: list[dict[str, Any]] = [
    dict(
        name="Pinocchio",
        node="pinocchio",
        cat=CAT_DYN,
        repo="https://github.com/stack-of-tasks/pinocchio",
        lic="BSD-2-Clause（LICENSE：BSD 2-Clause，CNRS / INRIA 版权）",
        last="2026-10-30",
        paper="README 引用软件论文：The Pinocchio C++ library（IEEE SII 2019，HAL hal-01866228）",
        article="语言 C++ + Python 绑定；许可 BSD-2；刚体动力学、FK/IK、雅克比、RNEA/ABA；Crocoddyl、Stack-of-Tasks 底层依赖",
        diff="一致。Crocoddyl README 确实说明用 Pinocchio 计算动力学与解析导数。",
    ),
    dict(
        name="Crocoddyl",
        node="crocoddyl",
        cat=CAT_DYN,
        repo="https://github.com/loco-3d/crocoddyl",
        lic="BSD-3-Clause（LICENSE：BSD 3-Clause，LAAS-CNRS / Edinburgh 版权）",
        last="2026-10-20",
        paper="Mastalli 等，Crocoddyl: An Efficient and Versatile Framework for Multi-Contact Optimal Control（ICRA 2020，arXiv:1909.04947，已核对标题）",
        article="C++；BSD-3；DDP 最优轨迹优化与 MPC；依赖 Pinocchio",
        diff="一致。README 描述为接触序列下的最优控制库，求解器基于 DDP，并用 Pinocchio 计算动力学与解析导数。",
    ),
    dict(
        name="SOEM",
        node="soem",
        cat=CAT_BUS,
        repo="https://github.com/OpenEtherCATsociety/SOEM",
        lic="GPLv3 + 商业双许可（LICENSE.md：Dual-licensed；商业产品「likely need to buy a license」；版权含 RT-Labs AB）",
        last="2026-10-14",
        paper="无论文；官方文档见仓库 README",
        article="C；许可 GPL；EtherCAT 主站协议栈，与伺服驱动器通信",
        diff="文章只写「GPL」，漏了关键约束：LICENSE.md 明确这是 GPLv3 + 商业双许可，用于商业产品多半需要购买商业许可。",
    ),
    dict(
        name="IgH EtherCAT Master",
        node="igh-ethercat-master",
        cat=CAT_BUS,
        repo="https://gitlab.com/etherlab.org/ethercat",
        lic="GPL v2（COPYING）；用户态库 LGPL v2.1（COPYING.LESSER）",
        last="2026-11-02",
        paper="无论文；官方站点 etherlab.org",
        article="C；许可 GPL；工业级 EtherCAT 主站，硬实时（Linux RT 补丁）",
        diff="仓库同时含 COPYING（GPLv2）与 COPYING.LESSER（LGPL v2.1），文章只写 GPL；硬实时依赖内核补丁来自文章，未在许可文件中核对。",
    ),
    dict(
        name="SimpleFOC（Arduino-FOC）",
        node="simplefoc",
        cat=CAT_BUS,
        repo="https://github.com/simplefoc/Arduino-FOC",
        lic="MIT（LICENSE：MIT，Antun Skuric 2020）",
        last="2026-10-04",
        paper="本次未核对论文",
        article="C++；MIT；BLDC/PMSM 的 FOC，电流/速度/位置环；STM32、ESP32、RP2040、Arduino",
        diff="许可一致；平台列表未逐项核对。",
    ),
    dict(
        name="ros2_control",
        node="ros2-control",
        cat=CAT_ROS,
        repo="https://github.com/ros-controls/ros2_control",
        lic="Apache-2.0（LICENSE）",
        last="2026-11-02",
        paper="官方文档 control.ros.org（本次未逐页核对）",
        article="C++；框架而非单库：HardwareInterface + Controller（joint_trajectory_controller 等）",
        diff="文章未写许可；仓库 LICENSE 为 Apache-2.0。",
    ),
    dict(
        name="MoveIt 2",
        node="moveit2",
        cat=CAT_ROS,
        repo="https://github.com/moveit/moveit2",
        lic="BSD-3-Clause（LICENSE.txt：BSD 3-Clause，Willow Garage）",
        last="2026-10-28",
        paper="Coleman 等，Reducing the Barrier to Entry of Complex Robotic Software: a MoveIt! Case Study（arXiv:1404.3785，已核对标题；为 MoveIt 1 时期论文）",
        article="C++；BSD；机械臂运动规划（OMPL）、碰撞检测、轨迹插值；偏路径规划，不是底层 S 曲线 OTG",
        diff="许可一致。",
    ),
]

NEW: list[dict[str, Any]] = [
    dict(
        slug="ruckig",
        name="Ruckig",
        cat=CAT_TRAJ,
        repo="https://github.com/pantor/ruckig",
        lic="MIT（LICENSE：MIT，Lars Berscheid 2021；README 徽章 MIT）",
        last="2026-09-29",
        paper="Berscheid & Kröger，Jerk-limited Real-time Trajectory Generation with Arbitrary Target States（RSS XVII 2021，arXiv:2105.04830，README 与 arXiv 均已核对）",
        tags=["trajectory-generation", "motion-control", "cpp", "real-time"],
        one="Ruckig 实时生成从任意初始状态到目标状态（位置、速度、加速度）的、受速度 / 加速度 / jerk 约束的时间最优轨迹，可在线重规划，C++ 为主并带 Python 绑定。",
        article="语言 C++（附 Python 绑定）；许可「社区版 MPL-2.0（商用免费），Pro 版闭源」；带 jerk 限制的时间最优轨迹，在线实时重规划、多轴 PTP、支持中间路径点；替代 Reflexxes；微秒级计算，硬实时友好。",
        primary="README：Ruckig 对「state-to-state」运动保证时间最优；允许给中间位置做路点跟随，此时联合计算路径及其时间参数化。",
        diffs=[
            "**许可不一致**：文章写社区版 MPL-2.0；仓库 LICENSE 与 README 徽章均为 **MIT**。",
            "**中间路径点不是实时的**：README 称社区版带中间路点时会切换到**云 API（非实时）**；要在自己硬件上实时计算需联系作者购买 Ruckig Pro。文章「支持中间路径点 + 硬实时友好」对社区版并不成立。",
            "README 提到 Pro 版有 `interrupt_calculation_duration`（微秒）等实时保证接口；「微秒级计算」的具体数字本次未复现。",
        ],
        related=[
            "../methods/trajectory-optimization.md",
            "../entities/ros2-control.md",
            "../entities/moveit2.md",
        ],
    ),
    dict(
        slug="plcopen-pyplcopen",
        name="pyplcopen / plcopen 运动控制内核",
        cat=CAT_TRAJ,
        repo="https://github.com/lusipad/plcopen",
        lic="Apache-2.0（LICENSE：Apache License 2.0；README「许可：Apache License 2.0」）",
        last="2026-11-02",
        paper="无论文；PyPI 项目页 pypi.org/project/pyplcopen（摘要：PLCopen motion control kernel — Python simulation facade）",
        tags=["plcopen", "motion-control", "cpp", "ethercat", "ot"],
        one="plcopen（PyPI 名 pyplcopen）是现代 C++17 header-only 的 PLCopen 风格运动控制内核，含时间最优 OTG、前瞻规划、轴与组状态机、功能块，以及 IEC 61131-3 ST 编译器 + 确定性字节码 VM；Python 包是其仿真门面。",
        article="语言 C++17 header-only、Python 封装；许可 MIT；PLCopen 运动内核（PTP、直线、圆弧、前瞻、连续插补、电子凸轮、同步、回零、轴组状态机），零分配实时周期；适用自研运动控制器、EtherCAT 上位机，纯算法不含总线驱动。",
        primary="README：C++17 header-only 运动控制内核；实时基础设施、时间最优 OTG 与前瞻规划、轴 / 组状态机、PLCopen 风格功能块与 ST 编译器 + 字节码 VM；周期路径零分配 / 零锁 / 无异常；是可嵌入库而非完整 PLC 运行时；PyPI 包为 `pyplcopen`，作者 lusipad。",
        diffs=[
            "**名称对应**：文章写的「pyplcopen」是 PyPI 包名，真实源码仓库是 **lusipad/plcopen**（PyPI 页面 Repository 链接指向它）；Python 侧是仿真门面，内核是 C++17。",
            "**许可不一致**：文章写 MIT；仓库 LICENSE 与 README 均为 **Apache-2.0**。",
            "**EtherCAT 尚未完成**：README 状态表写 EtherCAT「🔴 未开工」、是最大剩余软件块；文章把它列为「EtherCAT 上位机」适用场景，目前只能与外部 EtherCAT 主站搭配。",
            "README 写最新正式版为 v0.21.0（PyPI 页面仍显示 0.21.0）；项目较新、版本 0.x，文章未提成熟度。",
        ],
        related=[
            "../concepts/ethercat-protocol.md",
            "../entities/soem.md",
            "../entities/igh-ethercat-master.md",
            "../entities/ruckig.md",
        ],
    ),
    dict(
        slug="fastaccelstepper",
        name="FastAccelStepper",
        cat=CAT_TRAJ,
        repo="https://github.com/gin66/FastAccelStepper",
        lic="MIT（LICENSE：MIT，J.Kiemes 2020）",
        last="2026-10-20",
        paper="无论文；README 说明为 AccelStepper 的高速替代",
        tags=["stepper-motor", "motion-control", "embedded", "cpp", "esp32"],
        one="FastAccelStepper 是 AccelStepper 的高速替代库：为步进电机生成带加减速的脉冲，支持 AVR、SAM、SAMD51、ESP32 系列、Raspberry Pi Pico / Pico2 等，并（实验性）支持 Teensy 4.x。",
        article="语言 C++（嵌入式）；许可 MIT；步进电机 S 型加减速、脉冲输出，无浮点，适合 AVR/ESP32/Pico；适用单片机步进驱动、3D 打印机、小型传送机构。",
        primary="README：「a high speed alternative for the AccelStepper library」；平台表给出 ESP32 系列（MCPWM/PCNT、RMT、I2S）驱动的最大步频（如 200 kSteps/s）等。",
        diffs=[
            "**「S 型加减速」「无浮点」未核对**：README 的已读部分描述的是高速步进与线性加减速平台表，没有读到 S 曲线或「无浮点」字样；这两点按文章表述保留，未核实。",
            "许可 MIT 一致；平台支持范围以 README 平台表为准（文章仅列 AVR/ESP32/Pico）。",
        ],
        related=["../entities/simplefoc.md", "../concepts/field-oriented-control.md"],
    ),
    dict(
        slug="robotics-library-rl",
        name="Robotics Library（RL）",
        cat=CAT_DYN,
        repo="https://github.com/roboticslibrary/rl",
        lic="BSD-2-Clause（LICENSE.md；README「All source code files of RL are licensed under the permissive BSD 2-clause license」）",
        last="2025-04-15",
        paper="官方站点 roboticslibrary.org；本次未核对论文",
        tags=["kinematics", "dynamics", "motion-planning", "cpp", "library", "independent"],
        one="Robotics Library（RL）是自包含的 C++ 库，覆盖刚体运动学与动力学、运动规划与控制：空间向量代数、多体系统、硬件抽象、路径规划、碰撞检测与可视化，支持 Linux / macOS / Windows，BSD 许可、商用免费。",
        article="C++；BSD（商用免费）；运动学 + 动力学 + 碰撞检测 + 路径规划 + 可视化，独立库不强制 ROS；机械臂离线 / 在线规划，Windows + Linux 实时内核都支持。",
        primary="README：self-contained C++ library for rigid body kinematics and dynamics, motion planning, and control；spatial vector algebra、multibody systems、hardware abstraction、path planning、collision detection、visualization；用 CMake，可用 Clang / GCC / Visual Studio 编译。",
        diffs=[
            "许可：README 写明为 **BSD 2-clause**（文章只写「BSD」）。",
            "「Windows + Linux 实时内核都支持」：README 只说明可在 Linux / macOS / Windows 上运行，**未提及实时内核**；该点来自文章，未核实。",
            "仓库最近提交 2025-04-15，相对其他项目更新较慢。",
        ],
        related=[
            "../entities/pinocchio.md",
            "../entities/ompl.md",
            "../methods/trajectory-optimization.md",
        ],
    ),
    dict(
        slug="control-toolbox-ct",
        name="Control Toolbox（ct）",
        cat=CAT_DYN,
        repo="https://github.com/ethz-adrl/control-toolbox",
        lic="BSD-2-Clause（README：BSD-2 clause license；仓库根目录未见独立 LICENSE 文件）",
        last="2021-07-05",
        paper="Giftthaler 等，The Control Toolbox - An Open-Source C++ Library for Robotics, Optimal and Model Predictive Control（ICRA 2018，arXiv:1801.04290，已核对标题）",
        tags=["optimal-control", "mpc", "cpp", "eth", "auto-diff"],
        one="Control Toolbox（ADRL CT）是 ETH ADRL 的 C++ 库，用于机器人的建模、控制、估计、轨迹优化与 MPC。",
        article="C++；BSD；最优控制、MPC、自动微分、shooting 方法；适用非线性系统、机器人、飞行器最优控制。",
        primary="README：efficient C++ library for control, estimation, optimization and motion planning in robotics；求解器列表含 Classical Single Shooting、iLQR/iLQG、Multiple-shooting iLQR、GNMS、Direct Multiple Shooting 等；可用于大规模最优控制问题的 MPC 求解。",
        diffs=[
            "**维护状态**：README 顶部声明（2021 年 7 月）该库「currently only scarcely maintained」，对 bug 与需求响应缓慢；仓库最近提交 2021-07-05。文章未提，选型需注意。",
            "许可：README 写 BSD-2 clause；仓库根目录未找到独立 LICENSE 文件（文章写「BSD」）。",
            "「自动微分」：README 已读片段列出多种 shooting / iLQR 求解器；自动微分细节按文章保留，未逐项核对。",
        ],
        related=[
            MPC := "../methods/model-predictive-control.md",
            "../entities/crocoddyl.md",
            "../entities/pinocchio.md",
            "../entities/towr-trajectory-optimization.md",
        ],
    ),
    dict(
        slug="rtmotion-intel-eci",
        name="RTmotion（Intel ECI）",
        cat=CAT_BUS,
        repo=None,
        lic="文章写「开源」；本次未能定位公开仓库，许可未核实",
        last="",
        paper="Intel Edge Controls for Industrial（ECI）官方文档：PLCopen Motion Control（eci.intel.com/docs/3.3/components/plcopen-motion.html）与 RTmotion 文档（eci.intel.com/rt-motion-docs/index.html）",
        tags=["plcopen", "motion-control", "cpp", "ethercat", "intel-eci"],
        one="RTmotion 是 Intel 创建的开源 C++ 库，实现 PLCopen 运动控制标准第 1、2 部分；在 Intel ECI 中与 IgH EtherCAT Master 和 SoftPLC 配合构建通用运动控制应用。",
        article="语言 C++；许可「开源」；PLCopen 运动功能块，和 IgH EtherCAT Master 集成，适合工业自动化、AMR、机械手。",
        primary="ECI 官方文档（3.3）：「RTmotion is an open source C++ library created by Intel」，实现 PLCopen motion control standard 的 part 1 和 2，加上机器视觉与 ROS 2 扩展后已被部署用于离散自动化设备、AMR 与机械臂；通过 ECI 的 apt 仓库分发（含 `plcopen-motion`、`plcopen-servo`、`plcopen-ruckig`、`plcopen-databus` 等包）。",
        diffs=[
            "**未能定位公开 GitHub 仓库**：我尝试了常见仓库路径与搜索，没有找到可访问的 RTmotion 源码仓库；一手资料只有 Intel ECI 官方文档，因此**许可与最近提交均未核实**，本页不编造仓库地址。",
            "文档显示 ECI 另有 `plcopen-ruckig-dev` 包，即 RTmotion 体系可与 Ruckig 配合；该关系来自官方文档的包列表。",
        ],
        related=[
            "../entities/igh-ethercat-master.md",
            "../entities/plcopen-pyplcopen.md",
            "../entities/ruckig.md",
            "../concepts/ethercat-protocol.md",
        ],
    ),
    dict(
        slug="ompl",
        name="OMPL（Open Motion Planning Library）",
        cat=CAT_ROS,
        repo="https://github.com/ompl/ompl",
        lic="BSD（LICENSE：OMPL released as Open Source under the terms of a 3-clause BSD license）",
        last="2026-10-31",
        paper="Şucan、Moll、Kavraki，The Open Motion Planning Library（IEEE RA Magazine 2012，DOI 10.1109/MRA.2012.2205651，README 徽章给出）",
        tags=["motion-planning", "sampling-based", "cpp", "rrt", "prm"],
        one="OMPL 是开源的采样式运动规划库：提供 40 多种规划算法（RRT-Connect、PRM、KPIECE、RRT* 等）与 20 多种状态空间（SE(3)、欧氏空间等），可用 Python 与 C++ 扩展自定义规划器与状态空间。",
        article="C++；BSD；采样型路径规划（RRT、PRM 等），避障路径规划，不做平滑插补。",
        primary="README：over 40 sampling-based planning algorithms across more than 20 state spaces；easily extensible（Python 与 C++）；README 另提 VAMP 的 SIMD 加速规划可达毫秒级。",
        diffs=[
            "许可：LICENSE 写 3-clause BSD，与文章一致。",
            "「不做平滑插补」是文章对 OMPL 与 Ruckig / pyplcopen 的分工说法（路径规划 vs 轨迹插补）；README 描述的是规划，未讨论插补，该分工按文章保留。",
        ],
        related=[
            "../entities/moveit2.md",
            "../entities/robotics-library-rl.md",
            "../entities/ruckig.md",
        ],
    ),
]

SELECTION = [
    (
        "自研 EtherCAT 工业运动控制器（CNC / 机械手）",
        "pyplcopen（plcopen 内核）+ SOEM / IgH + Ruckig（可选）",
        "plcopen README 写 EtherCAT 未开工，需自行接主站；SOEM 商用需商业许可；Ruckig 社区版中间路点非实时",
    ),
    (
        "只需多轴平滑 S 曲线、jerk 限制轨迹",
        "Ruckig",
        "许可为 MIT（非文章所写 MPL-2.0）；带中间路点的实时计算需 Pro",
    ),
    ("机械臂动力学、力矩控制、DDP 优化", "Pinocchio + Crocoddyl", "一致"),
    ("单片机无刷 FOC 电机环", "SimpleFOC", "一致"),
    ("ROS2 机械臂项目", "MoveIt2 + ros2_control + Ruckig", "一致；ros2_control 为 Apache-2.0"),
    ("步进电机单片机", "FastAccelStepper", "S 型加减速、无浮点两点未核对"),
]


def _yl(items: list[str]) -> str:
    return "\n".join(f"  - {x}" for x in items)


def _entity(p: dict[str, Any]) -> str:
    rel = [*p["related"], f"../overview/{MAP}"]
    rel_lines = "\n".join(f"- [{r.split('/')[-1][:-3]}]({r})" for r in rel)
    diffs = "\n".join(f"- {d}" for d in p["diffs"])
    code = f"code: {p['repo']}\n" if p["repo"] else ""
    link = f"[{p['repo']}]({p['repo']})" if p["repo"] else "（未能定位公开仓库）"
    last = p["last"] or "未核实"
    summary = p["one"].replace('"', "'")
    src = p["slug"].replace("-", "_")
    return f"""---
type: entity
project_id: {p["slug"]}
tags:
{_yl(p["tags"])}
status: complete
updated: {TODAY}
{code}related:
{_yl(rel)}
sources:
  - ../../sources/repos/{src}.md
  - ../../sources/blogs/{BLOG}
summary: "{summary}"
---

# {p["name"]}

**{p["name"]}**（{link}）是 [Qt历险记《C/C++ 运动控制开源库分类整理》]({URL}) 中「{p["cat"]}」一类的项目。本页以**一手资料**为准，文章说法放在「文章说法核对」里对照。

## 一句话定义

**{p["one"]}**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OTG | Online Trajectory Generation | 在线轨迹生成 |
| PTP | Point-To-Point | 点到点运动 |

## 为什么重要

- 在文章的选型建议里属于「{p["cat"]}」；横向对比见 [{MAP_TITLE}](../overview/{MAP})。

## 一手资料

| 项 | 内容 |
|----|------|
| **仓库 / 官方来源** | {link} |
| **许可（核查）** | {p["lic"]} |
| **最近提交** | {last} |
| **论文 / 文档** | {p["paper"]} |

**一手描述：** {p["primary"]}

## 文章说法核对

- **文章：** {p["article"]}
{diffs}

## 关联页面

{rel_lines}

## 参考来源

- [{src}.md](../../sources/repos/{src}.md)
- [{BLOG}](../../sources/blogs/{BLOG})
- [原文（微信公众号）]({URL})
"""


def _repo_src(p: dict[str, Any]) -> str:
    link = f"<{p['repo']}>" if p["repo"] else "（未能定位公开仓库；见官方文档）"
    return f"""# {p["name"]} 一手资料

> 来源归档（repo / 官方文档）

- **标题：** {p["name"]}
- **类型：** repo
- **链接：** {link}
- **入库日期：** {TODAY}
- **线索来源：** [{BLOG}](../blogs/{BLOG})
- **许可（核查）：** {p["lic"]}
- **最近提交：** {p["last"] or "未核实"}
- **论文 / 文档：** {p["paper"]}
- **沉淀到 wiki：** [`wiki/entities/{p["slug"]}.md`](../../wiki/entities/{p["slug"]}.md)

## 一手描述（README / 官方文档）

{p["primary"]}
"""


def _blog() -> str:
    rows = []
    for p in REUSE:
        rows.append(
            f"| {p['name']} | {p['cat']} | [{p['node']}](../../wiki/entities/{p['node']}.md) | 复用 | {p['diff']} |"
        )
    for p in NEW:
        rows.append(
            f"| {p['name']} | {p['cat']} | [{p['slug']}](../../wiki/entities/{p['slug']}.md) | 新建 | {p['diffs'][0]} |"
        )
    body = "\n".join(rows)
    sel = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in SELECTION)
    return f"""# C/C++ 运动控制开源库分类整理

> 来源归档（blog / 微信公众号）

- **标题：** C/C++ 运动控制开源库分类整理
- **类型：** blog
- **作者：** Qt历险记（微信公众号，「qt」合集）
- **原始链接：** {URL}
- **发表日期：** 2026-11-02
- **入库日期：** {TODAY}
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信内置 UA），正文 2340 字符
- **一句话说明：** 把 C/C++ 运动控制开源库分成轨迹规划、机器人动力学 / 整机控制、工业总线与控制栈、ROS / ROS2 运动控制栈四类（文章开头称五类：轨迹生成、机器人动力学 / 控制、工业 PLCopen 运控、嵌入式电机底层、ROS 生态；正文实际按四个小节排列，加一节选型建议），标注语言、许可与适用场景。
- **说明：** 文章是转述式清单，**没有给出处**。本次对每个项目回到一手资料核对（仓库 LICENSE / README、官方文档、论文），并在各节点标出与文章不一致的地方。文章中的「WBC Whole-Body Control」只给了通用名称、没有仓库，不能对应到唯一项目，因此只链接概念页 [whole-body-control](../../wiki/concepts/whole-body-control.md)，不建节点。

## 项目 → 节点与一手核对（14 个有仓库 / 官方文档的项目）

| 项目 | 类别 | 节点 | 状态 | 与文章的主要差异 |
|------|------|------|------|------------------|
{body}

## 选型建议（文章）与核对

| 场景 | 文章建议 | 核对 |
|------|----------|------|
{sel}

## 对 wiki 的映射

- 地图：[{MAP_TITLE}](../../wiki/overview/{MAP})
"""


def _map() -> str:
    def row(name: str, node: str, cat: str, status: str, lic: str) -> str:
        return f"| {name} | {cat} | [{node}](../entities/{node}.md) | {status} | {lic} |"

    rows = [row(p["name"], p["node"], p["cat"], "复用", p["lic"].split("（")[0]) for p in REUSE]
    rows += [row(p["name"], p["slug"], p["cat"], "新建", p["lic"].split("（")[0]) for p in NEW]
    sel = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in SELECTION)
    rel = [f"../entities/{p['slug']}.md" for p in NEW[:3]] + [
        "../concepts/whole-body-control.md",
        "../concepts/ethercat-protocol.md",
        "../methods/model-predictive-control.md",
    ]
    return f"""---
type: overview
tags: [overview, motion-control, cpp, open-source, trajectory-generation, ethercat, ros2, wechat]
status: complete
updated: {TODAY}
related:
{_yl(rel)}
sources:
  - ../../sources/blogs/{BLOG}
summary: "Qt历险记 C/C++ 运动控制开源库清单的阅读坐标：14 个有仓库 / 官方文档的项目逐个回到一手资料核对（LICENSE / README / 官方文档 / 论文），新建 {len(NEW)} / 复用 {len(REUSE)} 个节点，并列出文章与一手资料不一致之处（Ruckig 许可、pyplcopen 许可与名称、SOEM 双许可等）。"
---

# {MAP_TITLE}

> **本页定位**：为 [C/C++ 运动控制开源库分类整理]({URL})（Qt历险记，2026-11-02）提供一手资料核对后的阅读坐标。文章是转述式清单、没有给出处，本页对每个项目都回到仓库 LICENSE / README 或官方文档核对；**以一手资料为准，文章说法仅作对照**。

## 一句话观点

**这份清单能用，但「许可」和「实时 / 可用性」两列不能直接信：Ruckig 与 pyplcopen 的许可写错，Ruckig 社区版带中间路点不是实时的，pyplcopen 的 EtherCAT 尚未完成，SOEM 商用需商业许可。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| OTG | Online Trajectory Generation | 在线轨迹生成 |
| PLCopen | PLCopen motion control standard | 运动控制功能块标准 |
| DDP | Differential Dynamic Programming | 微分动态规划 |
| MPC | Model Predictive Control | 模型预测控制 |
| FOC | Field-Oriented Control | 磁场定向控制 |

## 为什么单独做这张地图

- 文章把许可当「选型快照」给出，但其中有一手资料不支持的说法；运动控制库的许可（尤其 GPL / 双许可）与「社区版 vs Pro 版」直接影响能否商用与能否硬实时。
- 全库已有 7 个相关节点分散在各处，新增 {len(NEW)} 个后需要统一入口。

## 项目一览（许可为一手核对结果）

| 项目 | 类别 | 节点 | 状态 | 许可（核查） |
|------|------|------|------|--------------|
{chr(10).join(rows)}

> 「WBC Whole-Body Control」文章只给通用名、无仓库，不能对应唯一项目，请见概念页 [全身控制](../concepts/whole-body-control.md)。

## 文章选型建议与核对

| 场景 | 文章建议 | 核对 |
|------|----------|------|
{sel}

## 关联页面

- [全身控制](../concepts/whole-body-control.md)
- [EtherCAT 协议](../concepts/ethercat-protocol.md)
- [模型预测控制](../methods/model-predictive-control.md)

## 参考来源

- [{BLOG}](../../sources/blogs/{BLOG})
"""


def _patch_reuse(p: dict[str, Any]) -> None:
    path = ROOT / "wiki/entities" / f"{p['node']}.md"
    text = path.read_text(encoding="utf-8")
    head, sep, body = text.partition("\n---\n")
    if not sep:
        raise SystemExit(f"frontmatter 缺失: {path}")
    lines = head.split("\n")
    blog = f"../../sources/blogs/{BLOG}"
    page = f"../overview/{MAP}"

    def _append(key: str, item: str) -> None:
        if item in (x.strip().lstrip("- ").strip() for x in lines):
            return
        idx = next((i for i, x in enumerate(lines) if x.startswith(f"{key}:")), None)
        if idx is None:
            lines.extend([f"{key}:", f"  - {item}"])
            return
        if lines[idx].strip() != f"{key}:":
            raise SystemExit(f"{path}: {key} 为行内写法，需手工补")
        j = idx + 1
        indent = "  "
        if j < len(lines) and lines[j].lstrip().startswith("- "):
            indent = lines[j][: len(lines[j]) - len(lines[j].lstrip())]
        while j < len(lines) and lines[j].lstrip().startswith("- "):
            j += 1
        lines.insert(j, f"{indent}- {item}")

    _append("sources", blog)
    _append("related", page)
    lines = [f"updated: {TODAY}" if x.startswith("updated:") else x for x in lines]
    section = f"""## 一手资料核对（{TODAY}）

> 来自 [C/C++ 运动控制开源库清单]({URL}) 的交叉核对；以一手资料为准。

| 项 | 内容 |
|----|------|
| **仓库** | [{p["repo"]}]({p["repo"]}) |
| **许可（核查）** | {p["lic"]} |
| **最近提交** | {p["last"]} |
| **论文 / 文档** | {p["paper"]} |

- **文章说法：** {p["article"]}
- **核对结论：** {p["diff"]}

"""
    marker = "\n## 关联页面"
    if "## 一手资料核对（" in body:
        new_body = body
    elif marker in body:
        new_body = body.replace(marker, "\n" + section.rstrip("\n") + "\n" + marker, 1)
    else:
        new_body = body.rstrip("\n") + "\n\n" + section
    path.write_text("\n".join(lines) + sep + new_body, encoding="utf-8")


def main() -> None:
    txt = Path(sys.argv[1]).read_text(encoding="utf-8")
    for p in NEW:
        if (ROOT / "wiki/entities" / f"{p['slug']}.md").exists():
            raise SystemExit(f"节点已存在: {p['slug']}")
        for r in p["related"]:
            if not (ROOT / "wiki/entities" / r).resolve().exists() and not (
                r.startswith("../entities/") and r[12:-3] in {n["slug"] for n in NEW}
            ):
                raise SystemExit(f"关联页不存在: {r} ({p['slug']})")
    for p in REUSE:
        if not (ROOT / "wiki/entities" / f"{p['node']}.md").exists():
            raise SystemExit(f"复用节点不存在: {p['node']}")
    (ROOT / "sources/raw" / BLOG).write_text(
        f"# C/C++ 运动控制开源库分类整理\n\n> 原始抓取（{TODAY}，`scripts/fetch_wechat_article.py`），仅供溯源；来源：{URL}\n\n{txt}",
        encoding="utf-8",
    )
    (ROOT / "sources/blogs" / BLOG).write_text(_blog(), encoding="utf-8")
    (ROOT / "wiki/overview" / MAP).write_text(_map(), encoding="utf-8")
    for p in NEW:
        (ROOT / "sources/repos" / f"{p['slug'].replace('-', '_')}.md").write_text(
            _repo_src(p), encoding="utf-8"
        )
        (ROOT / "wiki/entities" / f"{p['slug']}.md").write_text(_entity(p), encoding="utf-8")
    for p in REUSE:
        _patch_reuse(p)
    print(f"Wrote blog + map + {len(NEW)} new entities, patched {len(REUSE)} reuse nodes")


if __name__ == "__main__":
    main()
