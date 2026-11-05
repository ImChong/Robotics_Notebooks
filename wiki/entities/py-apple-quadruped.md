---
type: entity
project_id: py-apple-quadruped
tags:
  - repo
  - independent
  - quadruped
  - low-cost
  - diy
  - open-source
status: complete
updated: 2026-11-01
code: https://github.com/ToanTech/py-apple-quadruped-robot
related:
  - ../entities/quadruped-robot.md
  - ../tasks/locomotion.md
  - ../overview/quadruped-open-source-projects-technology-map.md
sources:
  - ../../sources/repos/py_apple_quadruped.md
  - ../../sources/blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md
summary: "菠萝狗（Py-Apple Dog）是低成本、可 DIY 的大型全套四足机器人软硬件开源项目，主站聚合四个子项目：Dynamics 控制软件、Controller 万能控制器、Structure 机械结构、BLDC 无刷四足。"
---

# 菠萝狗 Py-Apple Dog

**菠萝狗 Py-Apple Dog**（[GitHub](https://github.com/ToanTech/py-apple-quadruped-robot)）来自 [古月居转载的四足机器人开源项目综述](https://mp.weixin.qq.com/s/pRGoPm41qYCNyHPWR6Dl6A)，属「其他」组。

## 一句话定义

**菠萝狗（Py-Apple Dog）是低成本、可 DIY 的大型全套四足机器人软硬件开源项目，主站聚合四个子项目：Dynamics 控制软件、Controller 万能控制器、Structure 机械结构、BLDC 无刷四足。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BLDC | Brushless DC Motor | 无刷直流电机 |

## 为什么重要

- 综述把它放在「其他」这条技术路线下，作为可读源码的参照项目；横向对比见 [四足机器人开源项目技术地图：从极简运动学到研究级硬件（36 个 GitHub 项目）](../overview/quadruped-open-source-projects-technology-map.md)。
- 文中星标：1224（文章表述，入库日未复核）。

## 核心信息

| 项 | 内容 |
|----|------|
| **仓库** | [ToanTech/py-apple-quadruped-robot](https://github.com/ToanTech/py-apple-quadruped-robot) |
| **最近提交** | 2022-04-11 |
| **可达性** | `git ls-remote` 可达（2026-11-01） |
| **文章归类** | 其他 |

## 文章说法核对

- **文章：** 文章仅在附录清单列出（1224 星）。
- **README / 核查：** README 列出四个子项目（py-apple-dynamics、py-apple-controller、py-apple-structure、py-apple-bldc-quadruped-robot）及性能 / 遥控 / 平衡 / 二次开发 / AI 功能表。
- **差异与保留：** 本节点只对应「主站」仓库；四个子项目未单独建节点（文章未点名）。最近提交 2022-04-11。

## 源码运行时序图

本仓库为聚合主站，可运行代码仅为 `Py Apple Dynamics V6.8/` 下的 MicroPython 程序（ESP32）；`Py-Apple Controller V4.0`、`二代菠萝（Structure V10.0）` 目录为 PCB、文档、结构件，无运行时代码。

入口：先烧录 `micropython.bin`，再烧录 `V6.8 源代码/` 下文件（`用前读我.md`）；上电后 MicroPython 运行 `main.py`，默认 `run_mode=0`（网页遥控）。

```mermaid
sequenceDiagram
    participant Main as main.py
    participant Web as web_c.py (线程, app_1)
    participant Timer as machine.Timer(1) 周期 10ms
    participant P as padog.py mainloop()
    participant Gait as PA_TROT.cal_t / PA_WALK.cal_w
    participant Att as PA_ATTITUDE.cal_ges
    participant Stab as PA_STABLIZE.stab (PA_IMU)
    participant IK as PA_IK.ik
    participant Servo as PA_SERVO.angle
    Main->>P: import padog（exec config.py 与 config_s.py，加载参数与舵机中位）
    Main->>Web: _thread.start_new_thread(app_1)，exec(web_c.py)
    Main->>Timer: t.init(period=10, PERIODIC, callback=loop)
    Web->>Web: socket 监听 80 端口，解析 control.html 发来的请求
    Web->>P: padog.move 或 m(spd,L,R)、gesture、stable、gait、参数赋值
    loop 每 10ms 定时回调 loop(t)
        Timer->>P: padog.mainloop()
        alt gait_mode==0 (TROT)
            P->>Gait: PA_TROT.cal_t(t,0,spd*10,h,L,L,R,R)
        else gait_mode==1 (WALK)
            P->>Gait: PA_WALK.cal_w(CG_X,CG_Y,l,...)
            Note over Gait: 内部读 IMU 并调用 padog.gesture 调整重心
        end
        Gait-->>P: 足端 x、y 偏移 P_
        P->>Att: PA_ATTITUDE.cal_ges(PIT_S,ROL_S,l,b,w,X_S,R_H)
        Att-->>P: 姿态/高度对应的足端基准 ges_x、ges_y
        opt spd、L、R 均为 0 且 key_stab 为真
            P->>Stab: PA_STABLIZE.stab() 更新 padog.PIT_goal / ROL_goal
        end
        P->>IK: PA_IK.ik(ma_case,l1,l2,P_+ges)
        IK-->>P: 4 条腿的大腿、小腿角度
        P->>Servo: servo_output → PA_SERVO.angle(通道 4..11, 角度)
    end
```

图依据：py-apple-quadruped-robot@d2bb34f，`README.md`、`Py Apple Dynamics V6.8/.../用前读我.md`、`V6.8 源代码/` 下 `main.py`、`padog.py`、`web_c.py`、`PA_STABLIZE.py`、`PA_SERVO.py`（头部）、`PA_IK.py`（头部）、`PA_WALK.py`（头部）；`PA_TROT.py`、`PA_ATTITUDE.py`、`PA_IMU.py` 内部实现未读，仅按 `padog.py` 中的调用画出。`run_mode=1/2/3`（OpenMV 串口模式）未画。

## 关联页面

- [quadruped-robot](../entities/quadruped-robot.md)
- [locomotion](../tasks/locomotion.md)
- [quadruped-open-source-projects-technology-map](../overview/quadruped-open-source-projects-technology-map.md)

## 参考来源

- [py_apple_quadruped.md](../../sources/repos/py_apple_quadruped.md)
- [wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md](../../sources/blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md)
- [原文（微信公众号）](https://mp.weixin.qq.com/s/pRGoPm41qYCNyHPWR6Dl6A)
- 源码核对（2026-11-02）：ToanTech/py-apple-quadruped-robot@d2bb34f，读过 README.md、Py Apple Dynamics V6.8 下的 用前读我.md、main.py、padog.py、web_c.py、PA_STABLIZE.py，以及 PA_SERVO.py、PA_IK.py、PA_WALK.py 的开头部分；时序图依据以上文件，未运行代码。
