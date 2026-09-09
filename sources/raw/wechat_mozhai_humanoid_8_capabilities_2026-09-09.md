# 原始抓取：一文讲透人形机器人 8 个关键能力：感知、抓取、全身控制、平衡、VLA、世界模型、数据、仿真

> 抓取方式：WebFetch（`mp.weixin.qq.com` 直拉正文）；`--no-images` 等价（未本地化配图）
> 入库日期：2026-09-09

- **标题：** 一文讲透人形机器人 8 个关键能力：感知、抓取、全身控制、平衡、VLA、世界模型、数据、仿真
- **作者：** 猫先生M（微信公众号「魔方AI空间」）
- **原始链接：** https://mp.weixin.qq.com/s/-33IGrRqnxM6ALuI5ynODg
- **系列：** 【从零走向 AGI】（项目页 https://ai-mzq.github.io/From-Zero-to-AGI/ ）
- **姊妹篇（文内自述）：** 具身智能 7 概念（VLM/VLN/VA/VLA/WM/WAM/VLX）、机器人数据 8 概念、RT-2→π0.7 技术演进、2026 具身智能技术栈

---

在小说阅读器读本章

【从零走向AGI】旨在深入了解通用人工智能（AGI）的发展路径，从最基础的概念起，逐步构建完整的知识体系。

项目地址🔗：https://ai-mzq.github.io/From-Zero-to-AGI/

欢迎关注【魔方AI空间】👇

前面几期文章，我们介绍了具身智能的7个核心概念：VLM、VLN、VA、VLA、WM、WAM、VLX，以及机器人数据 8 个核心概念：遥操作、示范学习、动作块、仿真、合成数据、Sim-to-Real、数据飞轮、失败回放等。

本文聚焦于人形机器人的 8 个关键能力：感知、抓取、全身控制、平衡、VLA、世界模型、数据、仿真，旨在建立一张直观的人形机器人技术能力地图。

让一台人形机器人把桌上的红色杯子放进水槽——需要看清杯子、判断抓取、弯腰防重心前冲、遇人避让等。Demo 只展示几秒流畅动作；难的是把理解转化为连续、稳定、可恢复的物理行动。

## 八大关键能力概览

| 能力 | 它要解决的问题 | 最直白的理解 |
| --- | --- | --- |
| 感知 | 环境里有什么、在哪里、状态怎样 | 认出杯子，还要知道从哪里抓 |
| 抓取 | 如何稳定接触和操作 | 手要贴准、抓稳，还要会纠错 |
| 全身控制 | 手、腰、腿如何协同 | 伸手够物体时，整个人都在动 |
| 平衡 | 扰动和负载下如何不摔 | 每一步都在实时找回稳定 |
| VLA | 视觉和语言如何变成动作 | 听懂"收拾桌面"，并开始干活 |
| 世界模型 | 动作会带来什么后果 | 先判断会不会撞、会不会洒 |
| 数据 | 如何积累真实经验 | 机器人经历过什么，决定它会什么 |
| 仿真 | 如何低成本训练与验错 | 在数字世界里先试更多次 |

基础能力：感知、抓取、全身控制、平衡。
智能中枢：VLM、VLA、世界模型、任务规划。
工程底座：数据、仿真、后训练、安全约束。

代表工作与系统示例：RT-2、Open X-Embodiment / RT-X、OpenVLA、Octo、π0 / π0.5 / π*0.6 / π0.7、Gemini Robotics、NVIDIA GR00T、Figure Helix、Tesla Optimus、Unitree G1 / H1 / H2、智元远征 A2 / 灵犀 X2 等。

## 1. 感知

面向行动的感知融合 RGB、深度、相机位姿、关节状态与惯性信息；关注三维位置、姿态、可接触区域、遮挡与物体状态。主动感知：看不清就靠近、不确定就换视角。Gemini Robotics 展示多模态理解、空间推理与执行结合。

## 2. 抓取

完整过程：接近、对齐、接触、力觉反馈、滑移检测、修正。触觉/力觉是接触后的第二套眼睛。RT-2 把动作 token 化与 VLM 共训；π0 用 flow matching 生成连续动作。

## 3. 全身控制

人形伸手时躯干、双腿、足底协同；逆运动学、轨迹优化、MPC、动力学约束与实时反馈仍是基础。分层：语言任务+VLA 规划 → 全身控制 → 关节级高频跟踪。

## 4. 平衡

IMU、编码器、足底力矩与实时控制；视觉可预判台阶/障碍，但毫秒级扰动不能等云端推理。负载变化会改变质心，步态须即时调整。

## 5. VLA

LLM 处理文本，VLM 关联图文，VLA 输出动作。RT-2 离散 token；近年走向子目标、主动视角、力觉修正与连续动作（扩散/flow）。Open X-Embodiment / RT-X、Octo、OpenVLA 为开源通用策略路线。

## 6. 世界模型

给定状态与候选动作，预测未来与风险；用于规划、失败恢复、合成数据与安全筛查。柔性物体、罕见接触、传感器误差与人介入会使预测偏离——适合预测环节，实时观测纠偏，硬安全约束兜底。

## 7. 数据

记录视觉、关节、动作、力觉与结果；长尾场景与失败恢复轨迹同样重要。DROID、Open X-Embodiment 说明质量、可复用性与边界覆盖；重复低质数据不如多样性与动作质量。

## 8. 仿真

批量改变场景、摩擦、噪声与扰动；Isaac Sim/Lab、RoboCasa、Habitat 覆盖不同环节。Sim2Real 需真实与虚拟反复校正：真机失败 → 仿真扩展 → 受控真机验证 → 新日志回训。

## 结尾

评估应看换物体/场地/干扰后的成功率、恢复能力、碰撞、延迟、能耗与安全距离——而非只看 Demo。

## 参考链接（文内）

1. https://ai-mzq.github.io/From-Zero-to-AGI/
2. Gemini Robotics：https://deepmind.google/blog/gemini-robotics-brings-ai-into-the-physical-world/
3. RT-2：https://arxiv.org/abs/2307.15818
4. π0：https://www.pi.website/research/pi0
5. Open X-Embodiment / RT-X：https://arxiv.org/abs/2310.08864
6. Octo：https://arxiv.org/abs/2405.12213
7. OpenVLA：https://arxiv.org/abs/2406.09246
8. DROID：https://droid-dataset.github.io/
9. Isaac Sim / Isaac Lab：https://developer.nvidia.com/isaac/sim
10. RoboCasa：https://robocasa.ai/
11. Habitat：https://aihabitat.org/
