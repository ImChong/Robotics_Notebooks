---
type: entity
tags: [humanoid, hardware, open-source, fourier, fftai]
status: complete
updated: 2026-05-18
related:
  - ./humanoid-robot.md
  - ./open-source-humanoid-hardware.md
  - ../overview/robot-open-source-wechat-issue01-curator.md
  - ../queries/humanoid-hardware-selection.md
sources:
  - ../../sources/blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md
summary: "傅利叶智能开源人形线 GRX N1：以 fftai.github.io 文档站与 FFTAI 组织 GitHub 为主入口；BOM/STEP 等物料常以网盘分发，需与官方渠道交叉校验。"
---

# 傅利叶 GRX N1（开源人形）

## 一句话定义

**Fourier GRX N1** 是傅利叶智能在 GitHub 组织 **[FFTAI](https://github.com/FFTAI/)** 下维护的开源人形软硬件栈之一；公开资料以 **[N1 文档站](https://fftai.github.io/fourier-grx-N1/)** 的安装、配置与控制 API 说明为主轴。

## 为什么重要

- **国产开源人形入口**：与商业整机 **GR-1** 等产品线对照时，N1 常作为「可 fork 的软硬件参考」讨论。
- **SDK 边界**：文档强调对整机应用与底层控制的封装，部署前应核对与你的控制栈（ROS2 / 自研中间件）的接口假设。

## 开源入口（策展摘录）

| 类型 | 链接 |
|------|------|
| 文档 / SDK | [fourier-grx-N1 文档站](https://fftai.github.io/fourier-grx-N1/) |
| 组织 GitHub | [FFTAI](https://github.com/FFTAI/) |

> 网盘物料（BOM、STEP、加工图、装机 SOP）在转载清单中常见；提取码与路径以 [微信原文](https://mp.weixin.qq.com/s/3mF9a5Iuk7-WMg4-jCfBtA) 为准，并以 **官方发布** 校验。

## 关联页面

- [人形机器人](./humanoid-robot.md)
- [开源人形硬件方案对比](./open-source-humanoid-hardware.md)
- [机器人开源宝库（微信策展第01期）索引](../overview/robot-open-source-wechat-issue01-curator.md)
- [Query：人形机器人硬件怎么选](../queries/humanoid-hardware-selection.md)

## 推荐继续阅读

- [傅利叶 GR-1 平台速览](./humanoid-robot.md#主流平台速览)（商业侧对照）

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| BOM | Bill of Materials | 物料清单，硬件零部件列表 |
| API | Application Programming Interface | 应用程序编程接口 |
| SDK | Software Development Kit | 软件开发工具包 |
| ROS 2 | Robot Operating System 2 | 机器人系统集成与通信的常用中间件 |
| SOP | Standard Operating Procedure | 标准操作流程，如渐进式真机验证 |

## 参考来源

- [wechat_jixie_robot_open_source_treasury_issue01_10_robots.md](../../sources/blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md)
