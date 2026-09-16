---
title: 代码真的能帮部署吗？FluxVLA、JEPLO与DIDO等12篇机器人论文资源整理
author: 具身智能小站
date: "2026-09-16 14:00:00"
source: "https://mp.weixin.qq.com/s/nsAslK7HCyhUaViGkSVgWA"
---

# 代码真的能帮部署吗？FluxVLA、JEPLO与DIDO等12篇机器人论文资源整理

今天筛选了**12 篇具身智能相关论文**，覆盖**VLA工程与部署**、**世界模型与扩散策略**、**灵巧操作与空间感知**、**人形与足式机器人安全**等方向。

**本期导读：**如果你只想抓重点，先看 **FluxVLA Engine** 和 **JEPLO**；一个代表从数据、训练到真机部署的工程化整合，一个代表在退化感知下仍可迁移的 LiDAR 足式运动。如果你做世界模型与闭环控制，**DIDO** 和 **WholeBodyWAM** 也值得放进跟踪列表；如果你在补机器人感知和系统可靠性基础，**SlotDiT** 和 **RobResilience** 更适合先读。

🔥 重点推荐

1. 算法到真机，中间缺的不是又一个模型

🔬 **FluxVLA Engine: A One-Stop VLA Engineering Platform for Embodied Intelligence**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguZWkPyL9InzzvnsIYlbiaAicrLdUNxxKbfoEWRDc1JibllRxbUdkxVZcYgbFuc3Uibicu75SAfVoA0OJ943YBtDCFzP6xm9vibAbFzzY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

📌 为什么读：

VLA研究常卡在数据格式不统一、训练栈难复用、评测口径不一致和真机接口各自为政。FluxVLA Engine把这些环节收进一个配置驱动的工程闭环，对要复现实验、切换模型或推进部署的团队，价值在于把“能跑论文”推进到“能审计、能迭代的系统”。

✨ 核心收获：

① **先统一契约再比较模型：**数据集、视觉语言／世界模型、动作头、奖励加权学习、仿真评测、推理后端和机器人 operator 通过共享接口连接，减少把工程差异误判成算法差异。

② **把人在环中纳入训练闭环：**支持 rollout、接管、纠正采集和奖励标注，并将离线学习、仿真验证、在线修正与真机执行串联起来，适合检查长程任务中的错误恢复路径。

③ **部署侧要看节拍与硬件边界：**论文摘要强调 RTC、加速推理、远程 GPU 服务和轨迹后处理；实际性能仍取决于具体模型、仿真任务、机器人接口与硬件配置。

📖 摘要精读：

VLA、WAM和离线强化学习不断扩展策略设计空间，但数据格式、训练框架、评测协议、推理运行时与本体接口的碎片化，仍阻碍稳定落地。FluxVLA Engine不再提出新的策略模型，而是以配置驱动的统一接口连接数据、训练、仿真、在线纠正、推理和真机 operator，并加入双臂组合仿真、自动数据生成和人在环反馈。作者将其定位为可复现、可审计的工程基座；其效果应结合具体模型和硬件条件理解。

💡 关键创新：

关键增量不是单一网络结构，而是**模型解耦的端到端工程合同**：同一套配置和数据／动作接口贯穿离线学习、仿真验证、在线修正与部署，并把 RTC、远程推理和人在环数据回流纳入统一流程。

🔗 论文地址：https://arxiv.org/pdf/2609.17210

🔗 开源代码：https://github.com/FluxVLA/FluxVLA

---

⚡ 值得关注

2. LiDAR不建图，也能让四足机器人看懂地形

🔬 **JEPLO: Joint-Embedding Predictive Learning for LiDAR-Based Legged Locomotion**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguYFBof5xwOJoBKv9KyQCHd1GenpoGAl2BicwoL8ibIiabjelCrjEiaYqrmBvpDxefTrADEicZaE4YMicbwDoN55aqbF3EGIejFibTR0VY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

📌 为什么读：

足式机器人在楼梯、箱体和遮挡环境中，感知退化往往比策略本身更先造成失稳。JEPLO用 proprio-exteroceptive JEPA 学习地形变化，配合教师—学生训练把 latent 表征接到运动策略，适合关注无图导航、sim-to-real 和轻量板载计算的团队。

✨ 核心收获：

① **用预测表征替代显式地图：**PE-JEPA从原始 LiDAR 与本体状态学习局部地形表征，再由 CJTS 管线训练受 latent 表征影响的 locomotion policy。

② **退化条件是重点评测维度：**论文报告其在遮挡、稀疏和噪声感知下保持更强鲁棒性，并完成多地形 sim-to-real；但部署依赖 Unitree Go2、Mid-360 LiDAR、IsaacLab／MuJoCo 与 Jetson 等具体条件。

🔗 论文地址：https://arxiv.org/pdf/2609.15770

🔗 开源代码：https://github.com/ASIG-X/JEPLO

---

3. 把多步视频生成压成一步，别把接触动态一起压没

🔬 **DIDO: Distilling Interaction-Centric Dynamics into One-Step Denoising for World Action Models**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguYicpFeNW4F0BOy9KdGaJbVQp6IHt9XqNU2fpVVxyTprDbIITkx1XU0KOwQzOuMdVibXbwXaQljtM1fKEdPbhjsjMMtImJ1GEDoY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

📌 为什么读：

WAM的迭代去噪延迟会直接侵蚀闭环控制频率，而简单截断又容易保留背景、丢掉夹爪与物体的交互动态。DIDO围绕这一反差做蒸馏，适合研究世界模型实时性、接触中心表征和长程操作泛化的读者。

✨ 核心收获：

① **蒸馏目标要盯住交互实体：**除分布匹配外，DIDO用夹爪、目标物及交互的边界框推理 token，并用 DINOv3 特征对齐目标物表征，避免一步生成只剩场景结构。

② **结果覆盖多种操作基准：**摘要报告 LIBERO、LIBERO-Plus 与 RoboTwin 平均成功率分别为 **99.0%**、**76.6%** 和 **92.0%**，另有真实机器人长程与泛化实验；这些数值仍是作者报告，需按任务口径复核。

🔗 论文地址：https://arxiv.org/pdf/2609.15570

🔗 开源代码：https://github.com/LoveJu1y/DIDO-WAM

项目页面：https://loveju1y.github.io/DIDO/

---

🧭 快速扫读

4. 让扩散模型在对象级 latent 里预测机器人未来

🔬 **SlotDiT: Object-Centric Representations for Diffusion Transformers**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguY3u4aXfWFStSSwAcWbhmEgkh1rBF4CtIwwcqHzmiac73LekmN5TibBcEGdkGLcUt2ZGqcvljfMkmEH18IMQ4Q7abppuwzEBJAWU/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

📌 为什么读：

像素或 VAE latent 对“哪个物体发生了什么”缺少显式结构。SlotDiT把场景分解为对象级 slots，再在统一 DiT 框架下比较 slot、VAE 与语义对齐表示，为机器人视频预测中的表示选型提供了一个清晰切口。

🔗 论文地址：https://arxiv.org/pdf/2609.17414

项目页面：https://slot-dit.github.io/

---

5. 机器人遭遇攻击，先判断还能不能安全运行

🔬 **RobResilience: Implementing and Evaluating a Resilience Framework for Cyber-Physical Embodied Systems**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUgubicMODd10dzjMLibvSVPSA6zKnsorcYbEZwnqAUiahcc18iaIibC2wibbTDiahL7oic8rNgqaW0Jrlfb777rw50vL8VvxPVQvkqtiaoVNc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

📌 为什么读：

检测到攻击并不等于系统知道该继续降级运行、切换缓解措施，还是立即停止。RobResilience在 Webots 的 PR2／ROS2 环境中运行时检查可容忍扰动、可容忍退化和缓解可行性，为具身系统安全状态机提供可复现实例。

🔗 论文地址：https://arxiv.org/pdf/2609.17349

🔗 开源代码：https://github.com/mahyamkashani/RobResilience

---

6. 让人工智能体先天带着父母差异出生

🔬 **Machine Zygote: Causal Biparental Heredity Before Learning in a Germline--Soma Artificial Agent**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguaZ5mXZzJMu3vibgQGsm846jrQMEf7sCWZ6gxwjAMClib0yREAuI4M9hghJWS0BFKKLnILpjxhK1ePzqoFJDsF3zwd92B3EzAVlg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

📌 为什么读：

如果智能体出生前就能继承行为倾向，如何把亲子相似与真正的因果遗传分开？Machine Zygote用双亲 germline 重组、冻结 soma 和干预实验，在不学习的模拟智能体中测试这一问题，并明确不外推到生物遗传或真实机器人。

🔗 论文地址：https://arxiv.org/pdf/2609.17300

🔗 开源代码：https://github.com/LyesSaadSaoud/machine-zygote

---

7. 人形全身操作，不必从零重学一遍

🔬 **WholeBodyWAM: Generalizing Pre-trained World-Action Priors to Humanoid Loco-Manipulation via WBC-Grounded Coordination**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgubd8SibfENBFYoibna1pWEXrniclYhBFA4K38EBA55lDPk2dn9SwfE0bic2gPEj06rGmBRM0CB3IYB4sVgcia8SPHXPicduB8rY6iaVSk/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

📌 为什么读：

现有 WAM 多聚焦桌面或单臂操作，人形机器人还要同时协调走路、全身控制和手部动作。WholeBodyWAM保留预训练世界—动作先验，再把异构 WBC 语义接入协调模块，探索从已有模型扩展到全身 loco-manipulation。

🔗 论文地址：https://arxiv.org/pdf/2609.16644

项目页面：https://wholebodywam.github.io/

---

8. 看不清手和物体时，用接近关系补回触觉线索

🔬 **ProxiDex: Learning Dynamics-Guided Proximity Policy for Dexterous Manipulation**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguYIFHrabU4ZZiceF8AuPxuK4uoRGzSvQRmyzj0bdeWP15oHolkknEwoRwotyWyicQKt0aoajqPt61WUBrQ34bAF7MvAHL6TPNHGE/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)

📌 为什么读：

灵巧手操作中，手部遮挡和触觉硬件差异让接触状态难以稳定观测。ProxiDex把手—物体接近关系转成硬件无关的交互表征，再学习动作条件下的接近动态，为视觉不可靠时的策略推理提供补充信号。

🔗 论文地址：https://arxiv.org/pdf/2609.16586

项目页面：https://proxidex.github.io/

---

9. 把性能策略和安全修正拆开学

🔬 **ResSafe: Learning Safety Filtering with Residual Reinforcement Learning for Humanoids**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguZ5hibO87daU8DNIrWSNvC75fcxhqqlDHyQnnKQJWcEIIImjoAj4SULxVVWeu1icqrK4TcYicD95dicP06TbTR9g6UNJjPPqSOqLZA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=8)

📌 为什么读：

一个策略同时追求任务表现、安全和抗扰，奖励权重常常难调。ResSafe让 nominal policy 专注任务，再由 residual policy 学习安全修正，把性能—安全权衡转成显式的两阶段设计，适合跟踪人形控制中的安全过滤思路。

🔗 论文地址：https://arxiv.org/pdf/2609.15988

项目页面：https://sciautonomy.github.io/ResSafe\_Web/

---

10. 传原始视频太贵，机器人通信只发任务相关语义

🔬 **Goal-Oriented Communications for Physical AI: Design and Testbed**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguZCWJXdjnAEWbK1G2Hv4ibmzv84THQOtp8fsRRpOXXGhIhw7nuqSEEH8RHBKQMic92w7y3OMzJR1Lu5iaLoLbsQtHWzD0WgbHp924/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=9)

📌 为什么读：

物理 AI 需要高频视频闭环，但原始图像流会同时挤压带宽、时延和边缘算力。本文把 3D 框、2D 场景图和 3D 场景图接入完整 5G—边缘—机器人链路，提供从语义提取到控制执行的实测视角。

🔗 论文地址：https://arxiv.org/pdf/2609.15895

项目页面：https://sites.google.com/view/goc-physical-ai-testbed

---

11. 用世界状态变化，给异构动作数据找共同语言

🔬 **WLA^3: World Latent Action Modeling for Semantics, Dynamics, and Kinematics**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguYG6ALspY0m49pxy7POaKvxia2QSW5EjiaD3leoZjwSFBU594ISGYR2nnnNibiaAib1lQgZicpibK6I8wlvibSjiaic90A9zk5YgxtU5sibPM/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=10)

📌 为什么读：

通用策略扩展受限于异构数据缺少统一、低噪声的动作监督。WLA^3从相邻世界状态变化中学习 latent action，并把同一表征复用于语义、动力学和运动学，让人类视频也能参与动作相关预训练。

🔗 论文地址：https://arxiv.org/pdf/2609.15870

项目页面：https://wla-3.github.io/

---

12. 深度不是额外通道，而要对齐到动作真正看的 patch

🔬 **StereoPatch: Patch-Aligned RGB-Depth Fusion for Spatial Perception in Robot Manipulation**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguaWFEJMMgKVBz99Czib5dUE0NmEXOqcAqgoDia3WZZe2uQWibn32A87ibiaia6kgUicsuyzEGPfu2FFeAjCLDHic38QqLCGficr1lgaTSmc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=11)

📌 为什么读：

外观相似的场景可能因目标高度、位置或接触几何不同而需要不同动作。StereoPatch把注册后的度量深度绑定到动作预测所用的 RGB patch，再用非对称 cross-attention 融合，针对控制相关的几何歧义做了轻量改造。

🔗 论文地址：https://arxiv.org/pdf/2609.15509

项目页面：https://aus.bot/research/stereopatch/

关注本号，持续跟进具身智能开源论文前沿动态。

整理不易，欢迎点赞、转发，留言交流。
