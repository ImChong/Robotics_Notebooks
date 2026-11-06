# Unitree G1 官方产品页规格通读

> 来源归档（原文通读 · 支撑 `wiki/entities/unitree-g1.md`）

- **标题：** Unitree G1 — Humanoid agent AI avatar（产品页与参数表）
- **机构：** 宇树科技（Unitree）
- **类型：** site / product spec
- **原文：** <https://www.unitree.com/g1/>（2026-11 经代理抓取完整 HTML，抽取全部可见文本通读；移动版 `/mobile/g1/` 内容同源）
- **一句话说明：** 官方给出的 G1 / G1 EDU 机械、关节、电气、感知、算力、电池与脚注限制参数，以及「模仿 + 强化学习驱动、OTA 持续升级」的产品定位。
- **核对范围：** 通读产品页全部文本：营销段落、完整参数表（机械尺寸 / 自由度 / 关节活动范围 / 电气特性 / 配件 / 其他）及 [1]–[9] 全部脚注与民用声明。页内视频与动图未观看；SDK 开发手册（support.unitree.com 返回 567，未能访问）与 GitHub 仓库页（403）未读，二次开发接口细节仅依赖 wiki 与 [unitree_ros2 归档](../repos/unitree_ros2.md)。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **定位：** "Humanoid agent AI avatar"，起售价 US $13.5K（不含税与运费），G1 EDU 需联系销售；强调 OTA 持续升级、"Imitation & reinforcement learning driven"、UnifoLM（Unified Robot Large Model）。
- **尺寸与重量：** 站立 1320×450×200 mm，折叠 690×450×300 mm；含电池约 35 kg（EDU 为 35 kg+）；小腿 + 大腿长 0.6 m，臂展约 0.45 m。
- **自由度：** G1 共 23 DoF；G1 EDU 23–43 DoF。单腿 6、腰 1（EDU 可选再加 2 个腰自由度）、单臂 5；EDU 可选三指灵巧手 Dex3-1（单手 7 DoF，拇指 3 / 食指 2 / 中指 2，可选触觉阵列）及额外 2 个腕自由度。
- **关节：** 工业级交叉滚子轴承；低惯量高速内转子 PMSM；全关节中空走线；双编码器；局部风冷。最大膝关节扭矩 G1 90 N·m、EDU 120 N·m；臂最大负载约 2 kg（EDU 约 3 kg，随伸臂姿态变化很大）。
- **活动范围：** 腰 Z ±155°（EDU 另有 X/Y ±30°）；膝 0–165°；髋 P ±154°、R −30~+170°、Y ±158°；EDU 腕 P/Y ±92.5°。
- **电气与感知：** 13 串锂电池，9000 mAh 速换智能电池，充电器 54 V 5 A，续航约 2 h；基础算力为 8 核 CPU，EDU 可选 Orin 等高算力模块；深度相机 + 3D LiDAR、4 麦阵列、5 W 扬声器、WiFi 6 / 蓝牙 5.2。
- **版本差异：** 二次开发（secondary development）仅 EDU 提供；保修 G1 8 个月、EDU 18 个月。
- **脚注限制：** 参数随场景与配置变化；部分演示功能仍在开发测试；结构复杂动力强，须与人保持足够安全距离；全球人形行业处于早期探索阶段，个人用户购买前应充分了解人形机器人局限；产品为民用，禁止危险改装与危险使用。

## 对 wiki 的映射

- [`wiki/entities/unitree-g1.md`](../../wiki/entities/unitree-g1.md)

## 参考来源

- 官方产品页：<https://www.unitree.com/g1/>
