---
title: 灵巧手、世界模型与人形控制：14篇论文，附公开实现与数据资源
author: 具身智能小站
date: "2026-09-11 14:00:00"
source: "https://mp.weixin.qq.com/s/yfJRh-sUy5PgqHWFExqECw"
---

# 灵巧手、世界模型与人形控制：14篇论文，附公开实现与数据资源

今天筛选了**14 篇具身智能相关论文**，覆盖**VLM/VLA机器人控制**、**世界模型与记忆规划**、**灵巧操作与示范学习**、**人形运动与具身评测**等方向。

**本期导读：**如果你只想抓重点，先看 **Show-Harness** 和 **IMLE-VLA**；一个讨论如何把通用 VLM 接到机器人动作空间，一个把 VLA 的迭代动作头压到单步推理。如果你做部署可靠性，**FARM** 值得放进跟踪列表；如果你在补灵巧操作与人形数据基础，**SEED-UMI** 和 **HuRo** 更适合深入。

🔥 重点推荐

1. 把 VLM 的“会说”变成机器人能执行的动作

🔬 **Show-Harness: Just a VLM Agent Can Play Robots**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgua06Vg9buXWgvQ69tz8HENQQryvat1SK9tgGMTGOz6xsVtgBGw4l2ENYFabKoyqjvRkWASw9UVz14ZBNZOv9Tibzg86t1iaLhNIA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

📌 为什么读：

许多 VLM 能理解任务，却难以稳定落到连续控制。Show-Harness 把机器人动作重写为模型容易推理的**离散语义动作单元**，再由具身解释器确定局部运动，让团队可以直接评估通用 VLM 的空间落地能力，并复用同一接口跨机器人部署。

✨ 核心收获：

① **接口本身就是动作空间：**动作单元保留语义名称，解释器提供步长和机器人相关落地；项目页还展示了去掉约定后成功率从高位跌到 **5%** 的对照。

② **零样本与轻量适配并行：**Gemini-3.1 Pro 等前沿 VLM 可零样本控制，Qwen3.5-2B 等小模型则通过少量微调进入低成本部署路径。

③ **资源入口完整：**仓库包含 harness、GUMI 浏览器示范采集、插件和训练流程；项目页同时提供模型与数据入口，适合从接口到真机复现。

📖 摘要精读：

论文提出 Show-Harness，用离散、增量式语义动作单元连接 VLM 意图与机器人运动，并由不同具身解释器确定实际步长。作者在跨任务、跨环境和跨具身测试中报告零样本与微调两条路线：跨任务为 **89%/86%**，跨环境为 **100%/88%**，跨具身为 **93%/87%**；仅在仿真训练的微调模型还能在真机完成 **13/20** 次，而两个可训练 VLA 基线为 0/20。

💡 关键创新：

新增的关键不是再训练一个 VLA，而是设计可被 VLM 直接操控、又能由解释器适配不同机器人的**语义动作接口**；GUMI 将同一动作空间延伸到无需专用遥操作硬件的示范采集。

🔗 论文地址：https://arxiv.org/pdf/2609.10522

🔗 开源代码：https://github.com/showlab/Show-Harness

项目页面：https://showlab.github.io/Show-Harness/

---

⚡ 值得关注

2. 让 VLA 从慢动作推理回到实时控制

🔬 **IMLE-VLA: Fast Single-Step Action Generation for Vision-Language-Action Policies**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguaZSqsP3z2e6QNM2FOasgg8Y5InIfwegRIr36Lf3awo7nYLiar79WyULsWoicLUbQEe6j3jkGG2tGDt5uNbybsmqRLH213hhtibzk/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

📌 为什么读：

VLA 的多步扩散采样会带来停顿、低频和较慢的任务完成。IMLE-VLA 用条件 IMLE 训练**单步动作生成器**，在保留多模态动作覆盖的同时减少采样循环，适合做 VLA 推理加速、动作平滑和真实机器人反应性评测。

✨ 核心收获：

① **先看控制频率，再看成功率：**在 L40S 上从 π₀.₅ 的 15 Hz 提升到 **55 Hz**；LIBERO 四套共 40 个任务、每任务 50 次评测下，平均成功率为 **98.0%**。

② **加速有代价边界：**H=30 时动作吞吐达到 **11.0×**，但更长开环窗口会牺牲重新观察的反应性；真机 Franka 四任务中作者报告 VLA 前向耗时下降 3.9–6.6 倍。

🔗 论文地址：https://arxiv.org/pdf/2609.10915

🔗 开源代码：https://kianhk6.github.io/IMLE-VLA/

---

3. 不另训监控器，也能读出策略快要失败了？

🔬 **FARM: Reading Failure Signals from the Internal Predictive States of a Frozen Robotic World Model**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguYZwl6RZxD03SYkpQ6Ae24G7m1tMtw4Pf5BJHSIBgDqNcuib7LVaPd6s9XdfwcKxibKSkN2BWd7oqaRibkrbQtl12hcZZ0gtw7KSo/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

📌 为什么读：

部署中的失败监控常依赖代理信号或额外大模型。FARM 追问冻结世界模型的预测状态里是否已经含有失败线索，并用一个轻量 readout 做跨任务、跨策略与跨平台检测，适合安全执行与低开销运行时监控。

✨ 核心收获：

① **先验证表示，再比较监控器：**仅训练 33,985 参数的 readout，在 7 个源任务的五折 OOF 评测中达到 **85.68/88.59** 的 pooled AUROC/AUPRC，优于低阶隐藏状态统计控制。

② **迁移不是自动成立：**作者在 PIPER X、SO-101 和 Franka 上测试固定 readout 与只更新 readout 的适配；严格未见任务仍明显更难，且方法需要内部世界模型状态和失败标签。

🔗 论文地址：https://arxiv.org/pdf/2609.11445

🔗 开源代码：https://github.com/HaoranPei-casia/FARM

---

🧭 快速扫读

4. 无人机变成地面机器人的“空中潜望镜”

🔬 **EVPeriscope: Extended Perception across Aerial and Ground Vehicles with Event-based Propeller Tracking**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguZo5BSWo0dv9Dk2AibSThb1iaG9FibYPlkkpQcgSGbzz6ricutFnXTg0H4GYv7RohWc5ZW9VGKR3WcUbVtocaf742ktRKkLXb4C8ib0/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

📌 为什么读：

地面机器人进入高草、夜间或强光环境后，自身传感器可能被遮挡或失效。EVPeriscope 利用地面机器人向上的事件相机追踪旋翼高频信号，完成无标记相对定位，并让无人机反过来提供扩展视角；田野测试覆盖最高 15 mph 风速与夜间条件。

🔗 论文地址：https://arxiv.org/pdf/2609.11920

项目页面：https://ongdexter.github.io/evperiscope

---

5. 用动作经验约束“未来会发生什么”

🔬 **UniMPA: A Unified Memory-Prediction-Action Model via Action-Grounded Transition Modeling**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguaF5lTKTFmdw1wudFlrtJgia4JYkAThownxWBo1mkKmm5IxgsKbeuh1icjvicUEMltYmWGNDjUmfus9EnWHg2H3XmS6pdK0TbAcNY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

📌 为什么读：

视觉相似的当前画面可能处于不同操作阶段，视觉上合理的未来也未必能被机器人执行。UniMPA 用持久进度流、转移关键像素流和两类视觉—动作记忆库，把未来预测与可执行动作原型绑定；在 LIBERO、RoboTwin 2.0、VLABench 及双臂真机套件上评估。

🔗 论文地址：https://arxiv.org/pdf/2609.11875

项目页面：https://JiuTian-VL.github.io/UniMPA-page/

---

6. 不给示范、不做仿真，手指怎样学会写字？

🔬 **Rapid Learning of Dexterous In-Hand Pen Writing through Real-Time Jacobian Estimation**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguYMknloBOnT4eFXZuWdxfnAZN68y95VHKLHOMkjkpg8vTZau4tt9uTI79mfyiajvXOhn4VCriaVwdVNEC2lzEAhiaSnNhEbThWfwQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

📌 为什么读：

接触丰富的手内操作往往需要复杂模型、仿真或大量示范。论文在真实 ORCA 灵巧手上在线估计手—物系统任务 Jacobian，用约 **18 秒** 激励初始化后开始写字，不依赖解析接触模型、仿真训练或预收集任务示范，并在空中和纸面轨迹上验证亚毫米级平面精度。

🔗 论文地址：https://arxiv.org/pdf/2609.11775

🔗 开源代码：https://srl-ethz.github.io/rapid-dexterous-writing/

---

7. 把人和机器人穿过的外骨骼变成同一把尺

🔬 **SEED-UMI: Sharing the Exoskeleton between human and robot for onE-to-one Dexterous demonstration**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguY93wxhJhRyN1EAic5q8Kfl4Cyr2pibGg13ibkx68P81XxuvPicePyeTLFlhOicXQeyksGwmuqOmWaia3UnwFx7BzFzo1StreU2or0Yw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

📌 为什么读：

灵巧手示范的难点不只是采集量，还在于接触动作能否忠实转到机器人。SEED-UMI 让人和机器人共享同一外骨骼，联合利用关节编码器与腕部相机做配对跨具身监督，在五项接触丰富任务上考察数据效率与策略质量。

🔗 论文地址：https://arxiv.org/pdf/2609.11753

项目页面：https://tengbo-yu.github.io/SEED-UMI/

---

8. 长任务记忆，不必每一步都塞进执行器

🔬 **Memory as Plans: World-Action Modeling with Memory-Grounded Planning**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgubGDRxjDe90cLqSicVAFI00g54XlPKVhBSSRCqLGorUxZ5Yo0fUouIDdAlln0xT0J0pVqMcXeOo1XLMibTwZG3gqupJ0zubft5FM/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)

📌 为什么读：

长程操作需要记住已经发生的视觉细节，但不断扩大执行上下文会推高延迟和显存。MaP-WAM 把多模态情景记忆转成分段语言—视觉计划，再由带进度预测的执行器按需切换，使执行器上下文长度保持固定；RMBench 与真机任务分别报告 83.3% 和 78.0% 成功率。

🔗 论文地址：https://arxiv.org/pdf/2609.11561

项目页面：https://sizhezhao.github.io/projects/MaP-WAM/

---

9. 深度传感器坏一半，人形机器人别立刻切盲

🔬 **CAP: Continuously Adaptive Perception-Blind Humanoid Locomotion via Learned Denoising**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguZBOcdPmiaEPPPIBhS2MgcJ6Fs79m7LfgvSO8OnL0LE7jS05EwqLmzkOzmdhNtKKzBRyD2Nw6iaz9IGBZfRruS5fIiaKCr7gYPEHo/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=8)

📌 为什么读：

真实地形中的深度输入会间歇遮挡、闪烁或出现户外伪影，感知策略与盲走策略的硬切换容易突然失稳。CAP 用去噪世界模型编码器和本体感觉变分编码器共同供给单一策略，并在训练中覆盖连续感知退化；Unitree G1 室内外测试验证了部分遮挡下的平滑退化。

🔗 论文地址：https://arxiv.org/pdf/2609.11553

项目页面：https://hoshi-no-ai.github.io/CAP/

---

10. 别只调初始噪声，直接改生成策略的中间表示

🔬 **Beyond Noise Steering: Dual-Latent Space Reinforcement Learning for Generative Robot Policy**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgub67ryia5ROWnbpHyIgPAqfiaH0FRBFaj14ASZkghic6gE45EtrGTD6nsaKFmdeQNtaJGicicV83CKXxbOq4IqBazx29ibDFfndZPoqI/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=9)

📌 为什么读：

生成式机器人策略的强化学习适配常只操纵初始噪声，难以控制去噪过程中的中间动作表示。DLSRL 在冻结生成器的前提下，同时学习噪声 latent 与动作表示 latent，并通过残差适配特征注入中间动作 token；RoboMimic 与 LIBERO 实验重点考察在线适配速度。

🔗 论文地址：https://arxiv.org/pdf/2609.11270

🔗 开源代码：https://github.com/xianchaoxiu/DLSRL

---

11. 让解析 IK 也能进入可微规划链路

🔬 **Planning along Differentiable Charts of Constraint Manifolds with the Inverse Function Theorem**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguafA0AoGMAbuIpDJXPwnq7JVWRTZeSnBS1uXPTJnqAicTgMX1EXkAQnnd1WdRU4PJgLNCLtlNHB4ajbK2AblZNjPMaWCgfhsRvE/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=10)

📌 为什么读：

带运动学等式约束的轨迹优化需要在约束流形上求梯度，而通用解析 IK 实现通常为速度优化，难以直接接入自动微分。论文用增广正运动学与逆函数定理，从黑盒 IK 参数化中恢复梯度，并在数值实验和下游运动规划任务中验证这条几何规划路径。

🔗 论文地址：https://arxiv.org/pdf/2609.10905

项目页面：https://cohnt.github.io/inverse-function-theorem-parameterization/

---

12. 真正危险的反应，要把每个动作执行出来测

🔬 **ReactHuman: A Physics-Grounded Benchmark for Human-Like Reactive Decision-Making in Embodied Multimodal LLMs**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgubO6sD9fs9xEBoYErLibUKQUibEicr1UZoYP0lqiadV9C9av3U1VLicQvj84zY0QQ8lfloWcEEI97Gh8mgg5Uay6CNDCoJQHguRVzBQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=11)

📌 为什么读：

只问模型“该接还是该躲”无法发现够不着、站位错误和冻结等物理后果。ReactHuman 把多模态模型放进突发家庭危险的模拟人形身体中，提供 **17 类事件、1000+ 可复现情景**与五项诊断指标，并用 240 Hz 刚体仿真生成无须人工标注的真值。

🔗 论文地址：https://arxiv.org/pdf/2609.10895

🔗 相关资源：https://huggingface.co/datasets/Alan123/reacthuman-benchmark-scaled

---

13. 钢琴机器人不只要弹对，还要弹出力度

🔬 **Expressive Robotic Pianist: Mastering Complex Piano Repertoire with Graph-Mimic and Musical Dynamics**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUgubZYs767picAzQAhOyeYGiaQTxWpp0SfvrOo0ic5t4MKdCYALYt473AJ4iaj7uuMG7oUoxdr9hQr2lath5eOhg9gRIf6IKeywbmMSY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=12)

📌 为什么读：

机器人钢琴演奏的难点从音符准确延伸到手指形态、连贯过渡和力度表达。论文用图结构模仿约束手指运动，再以物理启发的声学模型把击键速度对齐乐谱动态；UR5 加灵巧手系统在多种曲风与听众偏好测试中评估表现。

🔗 论文地址：https://arxiv.org/pdf/2609.10844

🔗 开源代码：https://github.com/yanhuhuhahei/Preference-ranking-statistics

---

14. 把海量人类视频改造成 VLA 的机器人经验

🔬 **HuRo: Robotizing Human Videos for Scalable VLA Pretraining**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguaK1qAa11Akoa4wbVy2iap4y5U4p46wtdrdhmq67qIoVrxKAsGvIEUb8OFcKibSXYqwFkicaz981YAmYWPic41G5oTfUhN3Svf7uoY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=13)

📌 为什么读：

真实机器人数据昂贵且难扩展，人类视频虽丰富，却缺少机器人视角与动作标签。HuRo 建立机器人化流水线，把异构人类视频转成机器人对齐的观测和重定向动作，并构建约 **63 万条 episode、1.42 亿帧**的数据集；四项真机操作任务显示，扩大机器人化预训练规模可提升整体与 OOD 完成率。

🔗 论文地址：https://arxiv.org/pdf/2609.10706

🔗 开源代码：https://3587jjh.github.io/HuRo/

关注本号，持续跟进具身智能开源论文前沿动态。

整理不易，欢迎点赞、转发，留言交流。
