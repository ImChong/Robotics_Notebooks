# PhoneBot 论文来源归档

## 元数据

- **题目：** PhoneBot: A Low-Cost Open Humanoid Robot Platform Reusing Smartphones
- **方法：** PhoneBot（智能手机作躯干计算与感知核心的开源小型人形）
- **作者：** Ruochen Hou, Quanyou Wang, Daniel Koh, Dennis W. Hong
- **单位：** 加州大学洛杉矶分校（UCLA）机械与航空航天工程系 RoMeLa 实验室
- **预印本：** https://arxiv.org/abs/2610.08737
- **HTML：** https://arxiv.org/html/2610.08737v1
- **版本：** v1，2026-10-06，cs.RO
- **项目页：** https://phonebot.dev（归档见 [phonebot-project.md](../sites/phonebot-project.md)）
- **开放状态：** 论文称软硬件完全开源；项目页现列硬件 CAD、BOM、STEP、打印件与装配教程，许可为 CC BY-NC 4.0。页面仍未提供可核验的软件源码仓库链接（核查日 2026-10-22）
- **入库日期：** 2026-10-12

## 论文要解决的问题

小型人形平台通常仍要独立的计算、视觉、通信与交互硬件，推高成本和复现门槛。PhoneBot 把消费级智能手机（IMU、相机、Wi-Fi、处理器、屏幕）放在躯干作主要感知与策略推理单元，下肢用低成本舵机，目标是面向教学与快速原型。

## 方法与训练信息

1. **机械：** 483 × 183 × 125 mm（H×W×D），总质量 1.8 kg（含手机、执行器、结构、电子与电池）；13 个 DYNAMIXEL XL430-W250-T（单价 $23.90，258.5:1 减速比，1.4 N·m 堵转扭矩，57 RPM 空载）；每腿 6 DoF（髋 yaw/roll/pitch、膝 pitch、踝 pitch/roll）+ 躯干 yaw 关节 J13（独立转动手机支架）；髋 pitch 范围不对称（J3 −90°~123°，J9 −123°~90°）。
2. **电子与通信：** 小型单板机驱动执行器总线；手机与机器人无线缆连接，指令/遥测/手柄输入走 Wi-Fi；Android 应用读 200 Hz IMU，组成 actor 观测，在 CPU 或 GPU 上运行 TensorFlow Lite actor（50 Hz），关节位置目标经 UDP/Wi-Fi 发给单板机。笔记本后端承担 whisper.cpp 语音识别、LLM 和 Piper 语音合成。视觉用 ML Kit / MediaPipe，SLAM 用 ARCore。延迟未报告。
3. **手机：** 测试四部 Android 机型：Honor 9（2017，二手）、moto g 2025、Galaxy A16 5G、moto g 2024；后三者为 $50–$90 预付费机型。
4. **运动控制：** MuJoCo/MJX 平地上的非对称 actor–critic PPO；actor 输入 52 维（仅手机可得信号），critic 输入 120 维（含仿真特权状态）；动作为 13 维关节位置残差（50 Hz），经含饱和、速度扭矩衰减、回差和摩擦的 XL430 扭矩模型转换；指令 vx, vy ∈ [−0.5, 0.5] m/s、ωz ∈ [−1, 1] rad/s，10% 站立指令；矢状面镜像在每批 PPO 更新中左右镜像数据，使每个物理步样本量翻倍；另有起身策略（恢复至 0.282 m 站高）与朝向策略（视线 yaw > 55° 开始迈步、< 30° 停止）。
5. **成本：** 13 个执行器约 $311（主要开销）；整机约 $400（不含手机，Table I）；无完整 BOM。

## 论文报告的结果

- **镜像消融（5 个种子，1.5×10⁸ 环境步）：** 回报上升更早、最终更高、种子方差更窄。0.5 m/s 前进、20 s 确定性展开：关节角对称误差 0.254 → 0.003 rad；左右扭矩不平衡 0.205 → 0.034；机械功率不平衡 0.339 → 0.040（无镜像 → 有镜像）。
- **手机 CPU 占用（空闲/IMU/策略/视觉/SLAM/HRI，%）：** Honor 9：1.1/12.1/9.8/17.9/未运行/2.5；moto g 2025：0.4/13.4/10.3/15.6/34.3/3.6；Galaxy A16 5G：0.8/8.5/7.1/17.6/28.3/5.2；moto g 2024：0.8/9.1/6.3/16.7/25.2/4.3。Honor 9 因无 Google 服务未跑 SLAM。
- **真机：** 行走、人体跟随、从俯卧起身、对话、拍摄、移动远程呈现均有演示，无定量指标；ARCore SLAM 行走时稳定仅为定性观察。

## 复现与解释边界

- 作者列出的局限：无手臂（不能操作）、仅平地训练、舵机扭矩限制负载与动态动作、对话依赖外部笔记本后端、当前栈受 CPU 限制。
- 正文内部不一致：确定性评估步数文中 5×10⁸、图 5 题注约 4.5×10⁸；Table I 高度 0.48 m 与正文 483 mm。
- 硬件资料入口与许可已由项目页确认（CC BY-NC 4.0）；论文中的软件开源声明仍缺少可核验的源码仓库链接，见项目页归档。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-phonebot-smartphone-humanoid.md
- 汇总页：wiki/overview/frontier-humanoid-quadruped-2026-10-05-10-09.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.08737>
- arXiv HTML：<https://arxiv.org/html/2610.08737v1>
- 项目页：<https://phonebot.dev>
