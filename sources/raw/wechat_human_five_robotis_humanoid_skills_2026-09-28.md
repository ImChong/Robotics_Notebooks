---
title: Humanoid Skills：从全身运动到操作任务
author: human five
date: "2026-09-28 19:22:18"
source: "https://mp.weixin.qq.com/s/JcaBxH1xBmRdEFjesOTC7A"
---

# Humanoid Skills：从全身运动到操作任务

![Image](https://mmbiz.qpic.cn/mmbiz_png/Kltic3d4ibvZicWiaIcmIA4RAnzXmzEugOcGCbVKCCI2xHDFAX3nLruqxEpFQ2yyuA5d52p8HawaJpGnjvLYpmLzRXdFsrQ6M0B3AYQVjl5XuPc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

Humanoid机器人需要掌握两类截然不同的技能：**全身运动（whole‑body motion）** 与**操作任务（manipulation）**。

在ROBOTIS，我们同时对这两个方向展开研究，NVIDIA软件已经成为我们开发与规模化落地上述技能的核心工具。

对于**AI Sapiens**项目，我们的**23自由度双足K1机器人**，机身高度**1355 mm**，重量**35 kg**，由**DYNAMIXEL‑Q QDD执行器**驱动；手臂具备5自由度，腿部6自由度，腰部1自由度，搭载板载计算单元**NVIDIA Jetson Orin NX**。每一项技能的执行都必须维持动态平衡，因此**NVIDIA Isaac Lab**提供了大规模仿真与强化学习RL环境，在部署到真实硬件之前完成全身运动技能的训练。

对于**AI Worker**项目，我们的**双臂半Humanoid机器人**搭载两套7自由度机械臂、四轮转向移动底座（swerve‑drive mobile base）、RGBD相机，板载计算单元为**NVIDIA Jetson AGX Orin**。该项目的核心挑战从动力学问题转变为数据问题：**NVIDIA Isaac GR00T 1.7**提供预训练机器人基础模型；而**NVIDIA Cosmos Transfer**通过合成增强技术拓展真实机器人演示数据的视觉多样性。

**核心见解**全身运动依赖可规模化仿真；操作任务依赖可规模化机器人数据。

本文完整介绍两套技术流水线：包含运动生成、运动重定向、强化学习RL、仿真到现实（Sim2Real）部署；也涵盖遥操作、GR00T微调、基于Cosmos的数据增强。 更重要的是，本文将介绍这套技术栈在ROBOTIS真实硬件上的实际表现：哪些能力可以成功迁移、哪些环节会失效，以及我们为适配流水线做出的修改。

阅读本文你将了解：

1. 端到端**全身运动模仿pipeline**如何将人类运动转化为可部署在真实Humanoid机器人上的策略。
2. 消除**Sim2Real鸿沟**，既需要能够精准复现真实机器人动力学的仿真器，也需要行为可预测、便于建模的硬件。
3. **NVIDIA Cosmos Transfer**如何对小规模真实机器人数据集做增强，提升经过微调的**NVIDIA Isaac GR00T 1.7**操作策略性能。
4. 为什么高接触操作任务仍是重大挑战，以及学习真实世界交互动力学，可能是实现Humanoid操作规模化的下一关键步骤。

## 两套Humanoid开发pipeline中的NVIDIA软件组件

Humanoid机器人开发不存在单一固定pipeline。全身运动与操作任务面临不同瓶颈，因此我们针对二者分别组合调用**NVIDIA物理AI软件栈**的不同模块。

![AI Sapiens whole-body motion pipeline: Kimodo and GEM-X, SOMA-Retargeter, Isaac Lab, Sim2Real](https://mmbiz.qpic.cn/mmbiz_png/Kltic3d4ibvZicYENxIK8Hf98vOT69kjnS1eibbCy6T4Ezd7pS1uzWbCgAMfFXFFuYNvruRCfOxUW2Y8ql2dK0IRy1XIDIHcSlDDL5rpWu8yQLk/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

AI Sapiens：Kimodo / GEM‑X → SOMA‑Retargeter → Isaac Lab → Sim2Real

![AI Worker manipulation pipeline: teleoperation, Cosmos Transfer, GR00T 1.7, evaluation](https://mmbiz.qpic.cn/mmbiz_png/Kltic3d4ibvZ8ZYTmRznuHCVqiaS63Ftf7kQbZTGQVSiaPibnzqAN4M5nXBvial2ic9viaYPSjTzBCIORrBkTjjml1uNslso6ufKQWsYibkG6ujSPia14/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

AI Worker：遥操作 → Cosmos Transfer → GR00T 1.7 → 效果评估

两套开发pipeline高层逻辑相似，但在不同阶段使用不同NVIDIA技术，覆盖人类运动表征、大规模仿真、机器人基础模型、数据增强、板载部署。

针对**AI Sapiens**：**NVIDIA Kimodo**与**NVIDIA GEM‑X**可以从文本、视频或者运动学约束条件生成三维人体运动。**NVIDIA SOMA Retargeter**将生成的人体运动映射到K1机器人本体；**NVIDIA Isaac Lab**提供大规模仿真环境，集成Cyclo Lab的BeyondMimic算法，学习跟踪参考运动。得到的策略随后迁移到AI Sapiens，依托DYNAMIXEL‑Q阻抗控制与**NVIDIA Jetson**边缘计算，在真实硬件上完成Sim2Real执行。

针对**AI Worker**： pipeline起始于真实机器人遥操作。**NVIDIA Cosmos Transfer**无需再次采集物理数据，即可提升演示样本的视觉多样性。整合后的数据集用于微调**NVIDIA Isaac GR00T 1.7**，最终得到的操作策略在真实机器人上执行板载推理。

两套pipeline目标一致：得到性能更优的机器人技能；只是调用NVIDIA软件栈的路径各不相同。

下文将完整拆解上述两套pipeline。

## 第一部分 — AI Sapiens：全身运动

整体目标分为四个阶段：获取人体运动、适配AI Sapiens机器人本体、在仿真环境训练策略、在真实硬件上以50 Hz复现运动同时保证机器人不会摔倒。

### 参考运动的生成与重定向

首要难点并非强化学习RL，而是把人体运动转换为机器人学习pipeline能够稳定读取的表征格式。

如果完全依赖动作捕捉采集运动，每新增一项技能，就必须重新开展一次数据采集工作。因此我们的前端可以接收多种运动源：视频、文本生成运动、已有的人体运动数据集、参数化人体表征，并且将全部输入转换为统一的参考格式。

![SOMA in Action: SOMA-shape, MHR, SMPL-X, Anny, and garment-measurement bodies driven by one skeleton](https://mmbiz.qpic.cn/sz_mmbiz_jpg/Kltic3d4ibvZicMl5HSIZwgxicnxK5at04aHXNqq6DFte5MzJ8w9P1CCgdcye91vXVlE8Xj4ohN4Au2jIvNpLm1qXffuXlkxDw66VArmFPeJTGo/640?wx_fmt=webp&animated=1&from=appmsg&watermark=1#imgIndex=3)

SOMA为多种人体表征提供统一骨骼模型

**SOMA**为本pipeline使用的统一人体运动表征。不同人体模型与运动源，都可以映射到同一套底层骨骼，为下游重定向模块提供标准化接口。

![Image](https://mmbiz.qpic.cn/mmbiz_png/Kltic3d4ibvZ8CQsyvjHmEpCGcnJ9NSmRQzCibv2H3Cy5LwHSfW7fEgU9ZlF9fUWwjQtb5TjargOQOVI13LBW1FSk5roFb6micAjnmbrnTCDUAw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

人体运动源 → SOMA → SOMA Retargeter → 机器人参考运动

处理流程从左到右依次执行。

1. **运动输入**：多种数据源均可接入pipeline。ROBOTIS当前支持GEM‑X与Kimodo两种运动生成方式，配套部署与使用手册可查阅对应文档。

- **视频 / GEM‑X**：从无约束视频恢复三维人体运动。
- **文本提示词 / Kimodo**：通过自然语言描述生成运动学动作。
- **SMPL / AMASS**以及**MHR / SAM 3D**：通过SOMA‑X接入已有的人形与运动表征。
- **BONES‑SEED**：提供立即可用的运动序列。

2. **统一人体运动表征**：SOMA把异构输入映射为统一骨骼表征。下游固定输入为：**77个关节旋转量 + 三维根节点轨迹**。
3. **机器人运动重定向**：人体运动无法直接复制给Humanoid机器人。人类和机器人的关节数量、运动范围、连杆比例、本体结构均存在差异。**SOMA Retargeter**将该问题建模为带约束重定向求解，把人体参考运动映射到23自由度K1机器人本体，同时满足机器人关节限位与运动学约束。
4. **参考运动输出**：输出`reference_motion.csv` / `.npz`文件，包含策略训练需要的23个机器人关节位置与根位姿。

这套标准化接口意义重大：运动生成模块与机器人学习模块完全解耦。只要上游新的运动源可以转换为该统一参考表征，下游强化学习RL pipeline就无需重新设计。

输入：视频、文本、运动数据集；输出：可直接用于机器人的参考运动。

### 在NVIDIA Isaac Lab中训练全身运动技能

NVIDIA Isaac Lab：4096个K1机器人实例，在随机化环境下并行学习同一套参考运动

完成人体运动向K1机器人重定向后，下一个挑战：如何把运动学参考轨迹转化成策略，使其复现运动同时维持平衡。

![Image](https://mmbiz.qpic.cn/mmbiz_png/Kltic3d4ibvZibcwlqGAGiaMsocsyKl1pYLzvTaNBUbMHATXOR14FGv5VPcHpQf5euoJR1QXSibQGt26ofx16wibQ7vlKLIDbH9jiaJiclVVMpMnNTw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

**NVIDIA Isaac Lab**正是全身运动pipeline的规模化引擎。

依托Cyclo Lab，我们并行训练**4096个K1仿真实例**，使用**BeyondMimic**运动跟踪框架。每一个仿真机器人跟踪同一套参考运动，同时通过域随机化引入细微物理条件差异。

| 参数 | 数值 |
| --- | --- |
| 并行环境数量 | 4096个K1实例 |
| 物理仿真频率 | 200 Hz |
| 策略运行频率 | 50 Hz |
| 域随机化内容 | 地面摩擦系数与恢复系数、默认关节偏移、躯干质心、速度扰动 |
| 墙钟训练耗时 | 约4小时（测试动作“Red Red Dance”） |
| 输出产物 | `policy.onnx`，`sim2real.yaml` |

*表1 全身运动跟踪任务的Isaac Lab训练配置*

物理仿真以**200 Hz**运行；策略运行频率为**50 Hz**，与真实机器人控制频率保持一致。更高的物理仿真频率可以为接触动力学提供足够分辨率；仿真与部署阶段策略频率完全对齐，消除Sim2Real不匹配的一项来源。

更关键的是，Isaac Lab允许策略同时体验成千上万种机器人本体与环境变体。策略不再只学习单一理想条件下的运动复现，而是学习跟踪参考运动，同时兼容接触条件、关节偏移、质量分布、外部扰动带来的各类变化。

#### BeyondMimic：学习跟踪参考运动

**[BeyondMimic：从运动追踪到通过引导扩散实现的多功能人形控制](https://mp.weixin.qq.com/s?__biz=Mzg5Mzg3ODEwNA==&mid=2247487861&idx=1&sn=2d326a0510b42b3bebae6a63d26e1686&scene=21#wechat_redirect)**

**BeyondMimic**将重定向后的运动转化为全身跟踪优化目标。在Cyclo Lab Mimic框架中，不同参考运动复用同一套跟踪公式。新增技能时只需要提供新的参考运动，不需要为每个技能重新搭建学习配置。

奖励函数包含三大核心目标：

1. **锚点与位姿（Anchor and pose）**：跟踪参考本体构型与运动；同时保留足够自由度，允许机器人调整接触状态维持平衡。
2. **平滑性（Smoothness）**：对过大扭矩、加速度、触碰关节限位施加惩罚，避免高频、会损伤机械结构的行为，这类行为往往很难迁移到真实硬件。
3. **接触安全性（Contact safety）**：惩罚非预期接触，鼓励物理可行的执行逻辑，防止策略在仿真中利用不真实的接触策略投机取巧。

**训练目标**

锚点与位姿 · 平滑性 · 接触安全性

不同参考运动可复用同一套目标函数。

标准化参考运动表征、BeyondMimic跟踪框架，加上Isaac Lab大规模并行仿真，整套训练pipeline可以复用于各类全身运动技能。

输入一份新参考运动，经过大约**4小时**训练，即可输出可部署文件组合：`policy.onnx` + `sim2real.yaml`

**核心见解**1份参考运动 → 4096个随机化仿真环境 → 1个可部署全身运动策略。

接下来需要验证：在数千仿真K1上表现良好的策略，部署到单台真实K1机器人上能否拥有一致表现。

### 在AI Sapiens硬件上部署

![Image](https://mmbiz.qpic.cn/mmbiz_png/Kltic3d4ibvZibbsd68ZZWZ5zwTEaicehXoWjC7Rty87ibsETuxBrrkibSibGhcUvXY8fRwLCcUrwNZZy9lA4VNGiaEqDF9AuZZbDtLwwq0ibyP751zs/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

仿真得到的全身策略部署至AI Sapiens机器人：舞蹈、匍匐、摔倒恢复、动态运动演示

在**NVIDIA Isaac Lab**完成训练后，学习得到的策略导出，通过Sim2Real直接部署到AI Sapiens。

部署pipeline保证观测结构、策略频率、关节映射与仿真环境保持一致；策略推理运行在**NVIDIA Jetson Orin NX**。

![AI Sapiens deployment pipeline: motion CSV, IMU, and joint encoders into policy inference, followed by 23 joint targets and DYNAMIXEL-Q impedance control](https://mmbiz.qpic.cn/sz_mmbiz_png/Kltic3d4ibvZ9UReaj7g3gf0Xd9Ryql37quSPFp4C7EE416npSWYyiaQcoSxbofliayTVAVruvrnpLUxia1WXSQzLC3UNVc8EvB9fOGichVQzPG3E/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)

运动参考 + 机器人状态 → 策略推理 → 23个关节目标位置 → DYNAMIXEL‑Q阻抗控制

处理流程从左向右执行。

1. **输入**：策略读取参考运动CSV文件，同时接收IMU、关节编码器实时测量数据；既获取目标运动，也读取机器人当前状态。
2. **导出策略**：训练得到`policy.onnx`和配套`sim2real.yaml`部署在**NVIDIA Jetson Orin NX**。策略推理运行频率**50 Hz**，与训练阶段保持一致。
3. **输出**：策略输出**23个关节目标位置**。仿真K1模型与实体机器人之间，关节顺序、缩放、偏移、坐标约定全部对齐。
4. **硬件执行**：关节目标位置交由**DYNAMIXEL‑Q阻抗控制**执行，将学习得到的策略转化为真实机器人柔顺全身运动。

**部署pipeline**NVIDIA Isaac Lab → NVIDIA Jetson Orin NX → DYNAMIXEL‑Q → 机器人真机

仿真与硬件可以保持一致的部署接口。真正的难点在于接口背后的动力学是否同样一致。

即使策略在大量仿真机器人中表现稳定，部署到实体机器人依然会行为异常：接触特性、执行器响应、摩擦、质量分布、传感器噪声、控制时延，每一项都会改变闭环动力学，使其偏离训练阶段仿真环境。

这就是**Sim2Real鸿沟**。

### 弥合Sim2Real鸿沟

我们首次硬件部署就遇到经典Sim2Real问题：策略在仿真环境完美跟踪参考运动，但相同行为无法直接迁移到实体机器人。

对于双足机器人，微小的模型失配会快速累积。接触行为、质量分布、执行器响应、摩擦、传感器噪声、控制时延，任意一项都会让闭环动力学偏离策略训练时见到的仿真环境。

**工程见解**弥合Sim2Real鸿沟需要双向改进： 让仿真更加贴近真实机器人；让真实机器人行为更加贴近仿真假设。

域随机化可以提升策略对仿真动力学不确定性的容忍度。但迁移效果同样取决于两点：基准仿真本身能否更好复现机器人；实体硬件本身行为是否足够可预测。

#### 仿真端：提升物理仿真保真度

![Image](https://mmbiz.qpic.cn/mmbiz_png/Kltic3d4ibvZibuM6icKNohCG9YBFua5cvgPsmiclru84k8EiaUQUufS1qwcCicZlDicdcicR0PAEMwmAzaySaN1fejdWmbu1jibKoPQOt2wXRghKKsSs/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=8)

机器人本体、整体学习pipeline完全不变，仅仅更换物理后端。

我们最初训练策略使用Isaac Lab默认物理引擎**NVIDIA PhysX**，仿真内可以很好跟踪目标运动，但部署后仿真与真机行为差异显著，尤其是平衡能力、地面接触响应。

随后我们将训练环境切换为**Newton**。Newton是开源可扩展物理引擎，基于NVIDIA Warp与OpenUSD构建；由NVIDIA、Google DeepMind、Disney Research共同开发，交由Linux基金会管理，用于推进机器人学习研发。更换仿真配置后，同一实体平台上运行得到的策略，行为**稳定性与一致性大幅提升**。

对于全身运动任务，地面断续接触本身属于任务的一部分。仿真接触动力学的质量，直接决定策略学习到的经验。

域随机化依然重要，但随机化和模型保真度解决的是不同问题：前者拓宽模型周围的分布范围；后者提升该分布所依托基准模型的准确度。

但这只解决Sim2Real问题的一半。

#### 硬件端：可预测的实时执行

![Image](https://mmbiz.qpic.cn/mmbiz_png/Kltic3d4ibvZibiauLKeJZUXUpOvtUQkNFsXicGBWXicHUdWibgvacZv5os88SaQSFqm9ibCicQXjic6qMm9XYLgI19dDNtSXjrC4AYR0WWOq61b86heE/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=9)

NVIDIA Jetson Orin NX与DYNAMIXEL‑Q构成AI Sapiens实时执行栈

优化仿真，让虚拟机器人逼近实体机器人。另一半工作，是让实体机器人动力学尽可能匹配训练阶段仿真假设。

| 子系统 | 规格参数 | 作用说明 |
| --- | --- | --- |
| 计算单元 | NVIDIA Jetson Orin NX | 机器人本地完成策略推理，不需要离线推理往返传输 |
| 控制循环 | 1 kHz，低时延 | 降低感知到执行的延迟 |
| 操作系统 | PREEMPT‑RT | 降低实时控制循环的调度抖动 |
| IMU | 8 Hz更新频率 | 为控制栈提供高刷新姿态估计 |

*表2 AI Sapiens实时执行栈硬件配置*

**NVIDIA Jetson Orin NX**直接在机器人本体运行训练策略；实时控制栈保证观测、推理、执行之间时序稳定、可预测。

但仅仅低时延，不能保证关节实际行为与仿真假设完全一致。

最后关键一环是执行器。

#### QDD架构对Sim2Real的意义

人们讨论Sim2Real鸿沟时，常常只把它当作仿真侧问题：**物理引擎可以多精确地复现真实机器人？**

反过来的问题同样至关重要：

**我们想要仿真的这台实体机器人，本身行为的可预测性如何？**

MIT风格阻抗控制器，期望关节输出扭矩写作：

该公式融合**位置误差**、**速度误差**、前馈扭矩，计算关节需要输出的扭矩。

- ：位置误差产生的扭矩
- ：速度误差产生的阻尼扭矩
- ：补偿重力或者已知动力学效应的前馈扭矩

想要控制器行为稳定可控，下发的命令扭矩必须在关节端产生可以合理预测的物理扭矩。

理想传动条件下：

也就是**关节输出扭矩应当与电机电流具备可预测的比例关系**。

- ：关节输出扭矩
- ：减速比
- ：电机扭矩常数，单位
- ：电机电流[A]

但真实执行器还会引入摩擦、回程间隙、传动损耗、反射惯量等各类非线性效应。这些效应扭曲指令扭矩与关节实际输出扭矩的映射关系。

这类不确定效应越强，仿真器越难复现真实关节动力学，**Sim2Real鸿沟也就越大**。

这正是**DYNAMIXEL‑Q的QDD架构**发挥价值的地方。

较低减速比、小回程间隙、高反向驱动能力、精准电流控制，降低指令执行器行为和实际关节响应之间的不确定性。更小减速比同时会降低电机反射惯量，反射惯量近似与减速比的平方成正比。

目标不是让真实执行器达到理想状态；而是让它的行为**具备一致性、可以被建模**。

策略迁移最终追求缩小两者差距：

强化学习RL训练阶段，策略与仿真机器人交互数百万次。仿真动力学越贴近实体执行器与控制系统的真实动力学，训练学到的经验，部署后就越有效。

**工程结论**仿真保真度提升 → 得到更精准机器人模型

QDD硬件性能提升 → 关节动力学行为更可预测 → 硬件更容易建模

弥合Sim2Real鸿沟，二者缺一不可。

## 第二部分 — AI Worker：操作任务

AI Worker项目面临的瓶颈发生改变。

全身运动任务，我们可以并行运行成千上万仿真机器人实现学习规模化。操作任务完全不同：策略需要感知物体、理解场景，在光照、外观、几何、物理交互变化的条件下稳定输出动作。

因此操作pipeline遇到的首要瓶颈是**数据**：高质量真实机器人演示样本价值很高，但规模化采集成本巨大。

### 采集操作演示数据

机器人学习数据可以想象成金字塔结构。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/Kltic3d4ibvZ9xUwyuJJSJQHOda5vgzaxSMoPLDmAe1422zNUAQxPnVichb4XL2LciaosAhkPgsAjqH4td6zjbsZvuVfdyZs45a1ibXPF9LtrPPc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=10)

机器人数据金字塔：底层是互联网规模人类数据；中层仿真与合成数据；顶层高保真真实机器人演示。

| 层级 | 数据来源 | 规模 | 机器人保真度 | 成本 |
| --- | --- | --- | --- | --- |
| 底层 | 人类视频、互联网公开数据 | 极大 | 低 | 低 |
| 中层 | 仿真、合成数据 | 大 | 中等 | 算力约束 |
| 顶层 | 真实机器人遥操作 | 有限 | 最高 | 硬件、人力、场地、时间成本 |

*表3 不同机器人数据源在规模与保真度之间的权衡*

金字塔顶端是**真实机器人遥操作**。采集得到的观测、动作最贴近部署阶段机器人会遇到的真实情况；但是每多采集一轮样本，都要消耗机器人运行时间、人力成本。

AI Worker平台支持三类遥操作设备采集演示样本：外骨骼设备、小型跟随主臂、VR设备。三者都可以输出高质量机器人运动轨迹，但全部受限于相同资源约束：**硬件 · 人力 · 场地 · 时间**

这带来根本性规模化难题：我们想要真实机器人演示样本的高保真特性，但不希望每一次算法迭代都要开展大规模物理采集。

预训练机器人基础模型、合成数据增强技术正是用来解决该问题。

我们开展的实验遵循简单流程：

1. 使用真实机器人演示样本微调**NVIDIA Isaac GR00T 1.7**，在AI Worker实体机器人上评估微调后策略。
2. 当发现由视觉样本覆盖不全带来的失效模式后，我们提出问题：

**不重复在实体机器人采集任务数据，NVIDIA Cosmos Transfer能否提升数据集视觉多样性？**

### NVIDIA Isaac GR00T 1.7微调

第一轮操作任务实验，目标将**NVIDIA Isaac GR00T 1.7**适配到AI Worker机器人本体，评估微调策略在真机的表现。

我们采集**268条遥操作episode**，约3小时真实机器人演示数据，用来微调**NVIDIA Isaac GR00T 1.7**。

| 参数 | 数值 |
| --- | --- |
| 真实机器人演示样本 | 268条遥操作episode（约3小时） |
| 基础模型 | NVIDIA Isaac GR00T 1.7 |
| 训练硬件 | NVIDIA RTX PRO 6000 |
| 训练耗时 | 约4小时 |
| 推理硬件 | 板载NVIDIA Jetson AGX Orin 32 GB |

*表4 AI Worker任务GR00T 1.7微调配置*

微调将预训练GR00T 1.7模型适配到**AI Worker本体、工作空间、演示样本对应的操作任务**。

得到的策略可以完成基础任务，我们以此作为真机基线，分析剩余失效行为。

真机评估发现一类系统性失效：

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/Kltic3d4ibvZibicOuPdAUMENFtBKzMibEsv2z7qccJDR6MpkLCjd5dC5GdrOyKqyr9aKZJmibyq7qftAvlibJjhMEWlMaPESCIreiaLSseA1D2icwGA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=11)

数据增强前：经过微调的GR00T 1.7策略偶尔会朝着夹爪投射的阴影运动，而不是朝向目标物体。

夹爪靠近桌面时会投射阴影落在目标物体附近；部分测试中策略把阴影视觉特征误判成目标。

根因判断是**训练分布的视觉样本覆盖不足**。演示样本采集的工作空间、光照条件相对固定，模型缺少样本区分物体有效特征和阴影这类场景特有线索。

直接的解决方案是在更多光照、场景条件下重新采集演示数据。但又会回到之前的瓶颈：**硬件、人力、场地、时间**。

于是我们尝试另一条路径：在保留已有机器人轨迹的前提下，提升训练集视觉多样性。

### 使用NVIDIA Cosmos Transfer拓展视觉多样性

我们使用**NVIDIA Cosmos Transfer 2.5**对已有真实机器人演示样本做数据增强。

Cosmos Transfer可以变换机器人交互录制片段的视觉外观，同时保留底层运动轨迹和任务结构。不需要改动原始机器人动作标签，就可以让策略见识更多样的视觉条件。

换言之：

**保留机器人运动轨迹；改变机器人看到的场景。**

![Image](https://mmbiz.qpic.cn/mmbiz_png/Kltic3d4ibvZ9VFz9hOxas5EdTiaYbDGbmVVfC0GImmJYQle0QMINFpIeoIwCHBdlfB8U5kb4oOiaeTtNE1IKsViblXaNUzSzARUZibvXyMT02vPA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=12)

NVIDIA Cosmos Transfer 2.5：基于原有真实机器人轨迹生成300份经过视觉增强的演示样本

基于原始**268条真实机器人演示episode**，我们通过Cosmos Transfer额外生成**300条增强视频**；合并后训练集一共**568条样本**。

| 数据集 | 样本数量 |
| --- | --- |
| 真实机器人原始演示 | 268 |
| Cosmos增强演示样本 | 300 |
| **合并训练集** | **568** |

增强的目的不只是单纯把数据集规模从268扩充到568。更重要的是，同一套操作动作对应的**观测样本分布得到拓展**。

重新训练之后，AI Worker真机测试观察到两点提升：

1. **抗阴影干扰能力提升**：策略更少把夹爪阴影识别为目标，更加稳定地关注物体本身。
2. **故障恢复行为改善**：抓取失败后，策略会重新靠近目标物体，而不是卡在原地。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/Kltic3d4ibvZibVhhJXhCjgveG1mibhLSM8bAhDH7yqD2nfUSojktaDL9zLoN8uSV14bcvPKo51SflvrzNDBQ3Jqm9ducgj7yJgWeavfCzMic9mo/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=13)

Cosmos增强+GR00T 1.7微调后：目标关注效果提升，抓取失败后具备重新趋近物体行为。

恢复行为并不是本次增强的直接优化目标，该现象属于实证观测，不算受控实验结论。一种可能解释：更丰富的视觉分布，促使策略更少依赖场景专属外观特征，更多学习任务相关视觉结构。

**实验结论**

268份真实演示 + 300份Cosmos增强演示 = 568训练样本

真机原始数据 → GR00T 1.7微调 → 真机暴露失效问题 → Cosmos数据增强 → 重新训练 → 鲁棒性提升

本实验中，**NVIDIA Isaac GR00T 1.7**作为预训练基座，适配AI Worker本体与任务；**NVIDIA Cosmos Transfer**不需要额外真机采集，拓展演示样本的视觉多样性。

## scaling Humanoid技能

![Image](https://mmbiz.qpic.cn/mmbiz_png/Kltic3d4ibvZ98BXTGgIY1MT7dLbBLVcUnakKFQtAD7qPelz1Y1ALlN2jibswicJ6Kj0T98L0wZLxVc6rxtOibd05rUPtG3yr86UVdQRGHGzyfqk/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=14)

整套pipeline落地后，我们对Humanoid技能规模化有了更清晰的理解。

针对**全身运动**：NVIDIA Isaac Lab提供试错的规模化仿真环境。成千上万仿真Humanoid可以反复体验各类接触、扰动、动力学情况，策略再迁移到实体机器人。

针对**操作任务**：NVIDIA Isaac GR00T 1.7提供性能强大预训练基座；NVIDIA Cosmos Transfer可以挖掘昂贵真机演示样本的更多价值，拓宽样本视觉多样性。

但同时我们看到下一个瓶颈。

Cosmos Transfer可以改变**机器人看到什么**，同时维持原有运动轨迹。它可以生成更丰富交互观测样本，但底层物理交互本身没有发生变化。

高接触操作任务最终需要理解更深层次问题：**机器人执行不同动作之后，世界会如何发生改变？**

抓取角度微小偏移就会造成物体打滑；一次推动会改变物体位姿；抓取失败需要一套完全不同恢复动作。想要规模化学习这类动作带来的结果，远比单纯扩充观测样本困难。

这正是World‑Action Models 极具研究价值的原因。WAMs在世界模型/视频模型基础之上，学习场景如何跟随机器人动作发生时序演化；为学习更丰富物理交互动力学提供潜在方向。

**公开挑战**全身运动已经拥有规模化试错载体：仿真环境。

高接触操作任务，仍然缺少一套规模化学习“机器人动作如何改变真实物理世界”的方案。

World‑Action Models是实现交互能力规模化的一条潜在路径，而不只是单纯扩充观测样本。

## 多技能融合

ROBOTIS正在推进研发，目标超越基础运动、基础操作能力。

移动能力层面，我们将**全身运动**向**全身移动（whole‑body locomotion）**拓展，让Humanoid机器人可以行走、摔倒恢复，在真实环境动态移动同时维持平衡。

交互能力层面，我们把基础操作任务升级为搭配**ROBOTIS Hand灵巧手**的**灵巧操作**，实现更加丰富、精准、自适应的物体交互。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/Kltic3d4ibvZicXP1DJAvl5eZ0vMt8XyaFjfqvv0La5we2zebFCpuFibXwKzzy7LRTkJ4plPO246XHNGSg55vq5V5gGJheBvcAiaOwPauZszBWyY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=15)

灵巧操作 + 全身移动 → Humanoid综合能力

依托NVIDIA软件栈，覆盖运动生成、仿真、机器人学习、数据增强、板载部署，两个方向的研发都得到加速。

我们下一步工作：把**全身移动**和**灵巧操作**整合到带灵巧手的Humanoid本体，朝着可以在真实环境移动、交互、完成实用任务的机器人迈进。

全身移动 + 灵巧操作 → 具备实用价值的Humanoid机器人
