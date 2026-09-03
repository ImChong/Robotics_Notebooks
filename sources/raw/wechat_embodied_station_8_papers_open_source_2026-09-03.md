---
title: 具身智能机器人最新开源论文速览
author: 具身智能小站
date: "2026-09-03 14:00:00"
source: "https://mp.weixin.qq.com/s/-UqboKHaoG5eu79u9XQU0w"
---

# 具身智能机器人最新开源论文速览

📅 2026年9月3日

### 👋 大家好！

❝

来了！2026 年新开始的一个系列，主要是整理具身智能领域最近发表的提供开源代码或数据集的项目(论文)，希望对相关领域的小伙伴有所帮助。获取这些论文的开源项目链接，可以直接在本文中查看。欢迎转发和关注！！👇

本文汇总 8 篇 最新**机器人**与**具身智能开源**论文，覆盖长视野**机器人**操作、3D-aware **World Action Model**、接触丰富 **HRC** 基准、**人形机器人**安全停止、**灵巧手**单示范泛化、**多视角** 3D 点跟踪、**测试时智能**综述，以及人机协作中的**多视角****证据融合**。整体来看，这一批值得关注的不只是模型名，而是**开源**/**项目入口**相对完整，方便圈内人继续复现、评测和二次开发。

**综述主线：****具身智能**正在从“看懂-出动作”的单步范式，转向更强的系统闭环：意图要能长期保持，**世界模型**要纳入 3D 几何，接触和安全要被物理一致地评估，部署期还要能用反馈、证据来源和恢复策略管理风险。

01 · arXiv:2609.02653

🔬 **HINT**: **Human-Intent** Inception for Long-Horizon Robot Manipulation

📌 Embodied AI · Long-Horizon Manipulation · **VLA** · Agentic Framework

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguYibgL9IhlZ6XDkQR6Riajyo0YiavMJYYjzgKunxc6JKINGRwGQEA72hBpbxBEw70xt4icWTibojttQCJibPwkDue3jquibQgAmUcP4rY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

✨ 在模式切换处稀疏推理意图，让**长视野**操作少走视觉捷径。

📖 人类能根据简单意图完成复杂操作，并随视觉观测变化持续调整；但当前 **VLA** 和动作策略在密集、动态视觉输入与稀疏语言指导下，容易让视觉相关性压过语义意图。**HINT** 是一个受人类操作原则启发的 **agentic framework**：只在 manipulation-pattern transitions 处调用**语义推理**来确定当前子任务和目标，再通过**多视角 grounding** 与**视觉跟踪**保持该承诺。论文探索 image-space semantic highlighting 与 **attention-prior** injection 两种视觉接口，在不向 foundation action model 引入额外可训练参数的前提下，把跟踪到的意图传给动作策略。三类**长视野**任务和 OOD 变体实验显示，**HINT** 在两个 foundation policies 上提升意图理解、任务进展和端到端成功率，同时保持**低延迟**控制。

💡 **长视野**操作的关键，是让语义意图不被短期视觉相关性带偏。

🔗 项目链接： https://robot-hint.github.io/

🔗 资料来源： https://arxiv.org/pdf/2609.02653

02 · arXiv:2609.02531

🔬 Spatially Aware **World Action Model** via Geometric Latent Diffusion

📌 Embodied AI · **World Action Model** · **3D** Geometry · Diffusion Policy

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguasJaOQea2XlFeYoYicE9tg8bS0UnqpiaZeSR8tWoEPBOOHDeWvZwevsmCL8Z8HbCHiaGN5yA7CeDRw3Pah8711j1ULOeDEAPQ7hY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

✨ 把 **RGB**-D 几何注入 **WAM**，让动作预测和未来状态预测同时 **3D**-aware。

📖 **World Action Model**s 利用大规模预训练视频扩散模型联合预测未来观测和动作，继承互联网视频中的视觉与物理先验，但主流模型仍主要依赖 **RGB**，缺少 **3D** 信息。**SA-WAM** 将预训练视频模型改造成同时预测 action、**RGB** 和 **depth** 的 Spatially Aware **World Action Model**，在单一 **diffusion backbone** 中实现 **3D**-aware world modeling 与动作预测。它使用非线性编码把无界 **depth** 信号映射到冻结 **VAE tokenizer** 期望的有界输入域，从而不需要 **3D**-specific fine-tuning 也能引入几何信息。论文报告 **SA-WAM** 在 **RoboCasa** 和 **LIBERO-Plus** 上达到 **SOTA**，并在 **UR5** 真实机器人随机环境评估中超过强 baseline，同时分析了世界模型预测质量与 rollout 成功率的相关性。

💡 **WAM** 要成为机器人底座，迟早要把 **3D** 几何放进预测闭环。

🔗 项目链接： https://jlopetegui98.github.io/projects/sa\_wam.html

🔗 资料来源： https://arxiv.org/pdf/2609.02531

03 · arXiv:2609.02402

🔬 A **Physics-Consistent** Benchmark for Contact-Rich Human-Robot Interaction in Assistive Care

📌 Embodied AI · Human-Robot Interaction · Contact-rich Benchmark · Assistive Care

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUgubGKq5ibfY7P5kNDc2BUzpHbwUOc8fHc0YSG1s4yk09ByArCO7lEFxNYAnia2RjdkufJmMAqXqQpF5nz2EpQxvxjqqCjCyoiaoOMs/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

✨ 把物理响应人体、力安全筛查和任务成功放进同一个 **HRC** 基准。

📖 常规任务级评估只问机器人是否完成指定动作，却可能漏掉真实人体接触中才暴露的失败；这对**接触丰富**的**辅助护理**尤其关键。论文提出一个 physics-consistent **benchmark**，并以机器人辅助洗浴为实例，结合可变形、被动响应的人体，任务成功之外的 physics-aware scores，以及冻结的 vision-only / scorer-only 评估协议。为建立物理有效性，作者用 **Franka** impedance pushes 在医疗护理人偶上采集 **force-indentation** 测量来校准局部仿真响应。在 T1-T7 冻结协议和每方法 140 次运行下，LLM-augmented state machine 的任务成功率为 **72.9%**，经正确区域与力安全筛查后降至 **56.4%**；**VoxPoser** 接触更轻更稳定但仅完成 27.9%；zero-shot **π0.5** 任务成功率为 0.7%。

💡 接触型 **HRC** 不能只看完成任务，还要看接触是否物理有效。

🔗 项目链接： https://anonymous.4open.science/r/Physics-Consistent-Benchmark\_4\_HRC-8DBF/

🔗 资料来源： https://arxiv.org/pdf/2609.02402

04 · arXiv:2609.02358

🔬 **Humanoid** Safe Stop via Learned Stoppability Value

📌 Embodied AI · **Humanoid** Robot · Safe Stop · Reach-Avoid

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguYAbICew2tseu4dztaCVHGxGNWZM8iawl0665roLicVJxOX1nicKbRABxnCgsHMIKwGKoIOpZpl8aMlXeiaK0muyPU9n5kHcibVnGMo/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

✨ 让**人形机器人**先判断“还能不能安全停”，再决定停或进入防摔策略。

📖 **人形机器人**响应急停命令时通常执行固定动作，却不判断从当前状态是否真的能安全停下。**Safe-Stop** 将紧急停止建模为 **reach-avoid** problem，提出与上游任务无关的框架：一个 learned **stop policy** 配合两个互补的 **stoppability** estimators。stop-probability estimator 由固定停止策略的实际结果监督，捕捉 learned controller 的涌现停止行为；**reach-avoid**ance estimator 则由物理状态上的 **Hamilton-Jacobi** backup 监督，提供可恢复性信号。由于停止策略和估计器不依赖急停前的行为策略，它们可迁移到多种上游任务。部署时，**Safe-Stop** 只有在两个估计器都认为停止可行时才提交停止，否则切换到以 **damping fallback** 实例化的 **fall policy**。

💡 急停不是固定反射动作，而是一个状态可恢复性判断问题。

🔗 项目链接： https://junfeng-long.github.io/safestop/

🔗 资料来源： https://arxiv.org/pdf/2609.02358

05 · arXiv:2609.01938

🔬 One Demonstration, Many Objects: Generalizing Manipulation via Local Contact Geometry

📌 Embodied AI · Dexterous Manipulation · Sim-to-real RL · Contact Geometry

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguaB9A5BlKicicbuO9STicrXiaODSp06ye2OIfc7Ct4cQ6sgicspfib9Zca8QvPxNMp0lH7ibeNFhD6puib9MibM4la0BJ0JmS1ABrxy7Og0/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

✨ 用**局部接触几何**泛化单个示范，让**灵巧手**跨物体迁移。

📖 多指机器人手有望实现接近人类的灵巧操作，但大规模**灵巧手**数据采集仍然困难；从人类示范学习是一种可扩展替代方案。近期 **sim-to-real** RL 方法虽会引入此类先验，却常缺少显式鼓励精确接触的 reward，或难以泛化到未见物体。**DemoMimic**（**DexterousMotionMimic**）提出让策略关注接触点附近的局部几何，并用 **contact-centric rewards** 鼓励精确接触、提升仿真到真实的一致性。由此得到的单个真实世界策略，可在局部接触结构保持一致时跨不同形状、尺度、质量和摩擦的物体迁移。真实消融实验显示，**DemoMimic** 在 **16 个物体**、**4 个任务**、2 种机器人手本体上达到 **71%** 成功率，并相比 baseline 具有最小 **sim-to-real** drop。

💡 **灵巧手**泛化不一定靠海量示范，局部接触结构可能更关键。

🔗 项目链接： https://demomimic.github.io/

🔗 资料来源： https://arxiv.org/pdf/2609.01938

06 · arXiv:2609.01899

🔬 **TAPVid-MV**: A Benchmark for Tracking Any Point in 3D Across Multiple Views

📌 Embodied AI · Multi-view Tracking · 3D Point Tracking · Benchmark

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguYNgG25WUyZpnRsekvwicmKianJXQibHHsicINiaf8EQ6tT7QoIGBPibEwk0PicrMKyBJlcIZ4jZibGTZ34rGb3icStG5yg2pjNAm4jQYIE/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

✨ 首个多同步视角、相机运动条件下的长时 3D 任意点跟踪基准。

📖 多相机系统在机器人、AR/VR 和自动驾驶中越来越实用，因为互补视角能减少深度歧义，并在遮挡下保持可见性；但现有点跟踪 **benchmark** 多聚焦单视频或静态多相机 rigs。**TAPVid-MV** 是面向该设置的首个 **benchmark**，包含 **284** 个序列、**1,142** 条标定相机流和 **109,769** 条点轨迹，覆盖 7 个子集，从室内外机器人、人类活动、驾驶到合成程序化场景。轨迹由传感器深度、**LiDAR**、**SLAM**/**SfM** 点、人类网格、物体网格和仿真等辅助模态获得，并由人工视觉验证。超过 30 个 baseline 的评估显示，目前没有方法接近解决该任务，**多视角**点跟踪器也未稳定优于单目点跟踪器；联合重建与跟踪评估进一步指出 **geometry recovery** 是准确 **3D 点跟踪**的主要瓶颈。

💡 **多视角**跟踪的短板，往往先卡在几何恢复而不是跟踪头。

🔗 项目链接： https://tapvidmv.github.io/https://tapvidmv.github.io/dataviewer

🔗 资料来源： https://arxiv.org/pdf/2609.01899

07 · arXiv:2609.01679

🔬 A **Survey** on Self-Improving **Test-Time Intelligence**: Feedback-Driven Adapting, Learning, and Scaling at Inference

📌 Embodied AI · **Test-Time Intelligence** · **Survey** · Feedback-Driven Learning

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguYhtJQvO12qvdQibzGN3rAmib4zemrkf3elft2W4mp4AGcd53Qxbfic6bC5NicvzJ3wicR2xlETaww7R9XzldnVYRTlNXvcwz17iacwA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

✨ 统一测试时适应、学习与扩展，梳理部署期自改进 AI 路线图。

📖 AI 系统在部署过程中改善自身行为的能力正变得越来越重要；推理不再只是固定模型的静态执行，越来越多工作研究如何利用 test-time 信息和额外计算即时优化行为。既有发展大体沿两条线展开：一类用测试时信号修改模型状态，另一类通过更多采样、工具使用等**推理期**资源提升预测，但二者常分属不同社区、术语割裂。该综述提出 feedback-driven **Test-Time Intelligence**（**TTI**）作为统一视角，关联 **test-time adaptation**、**test-time learning** 和 **test-time scaling**，强调它们的区别与在混合系统中的重叠。文章覆盖视觉、语言、多模态、生成模型、**机器人**和医疗等领域的方法范式、代表应用与开放挑战，并提供相关资源集合。

💡 **机器人**部署期自改进，需要把反馈、状态更新和推理扩展放在同一张图里。

🔗 项目链接： https://github.com/mr-eggplant/awesome\_test\_time\_intelligence

🔗 资料来源： https://arxiv.org/pdf/2609.01679

08 · arXiv:2609.01662

🔬 Not All Agreement Counts as Corroboration: Provenance-Conserving Multi-View Fusion for Typed Action Admission in Human-Robot Collaboration

📌 Embodied AI · Human-Robot Collaboration · Multi-view Fusion · Action Admission

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguZic6IPbSHKYcABYTC18NEffSdb8NeVhDBibxbHs1kAdcMIpzlkthKfib8hVADMv96zYL8ibtea5iaumudg4bsAwY80G4nLubwHp1cM/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)

✨ 区分计算重复与证据独立来源，让多视角融合更适合 **HRC** 动作准入。

📖 对具身系统而言，预测一致并不自动意味着证据足以支持行动，**证据来源**同样关键；对同一观测反复推理会放大一致性，却不增加新证据。**PACT** 将 evidence countability 视为 **provenance**-conserving fusion 与 **typed action admission** 中的关系变量：给定 **provenance** partition 后，它保留每个单元内共享的 coordinatewise support，只在不同可计数单元之间累积证据，并把未满足 release conditions 的情况映射为 **HOLD**、**CONFIRM** 或 **FALLBACK**。在 48 个场景簇的 **31,200** 次评估中，**PACT** 获得 **0.0861** ncsAURC；离线 **HRC** 设置中，camera-grouped **PACT** 在 60 个 episodes 里准入 57 个 **Qwen3-VL-32B** reference-consistent candidates 中的 47 个，未观察到 reference-inconsistent admission。

💡 多模型同意不是安全背书，只有来源可计数时才算佐证。

🔗 项目链接： https://github.com/ZekaiJ/PACT

🔗 资料来源： https://arxiv.org/pdf/2609.01662

**综合观察**

综合来看，这 8 篇论文的共同信号很明确：**具身智能**的竞争点正在从单一模型能力扩展到系统级可靠性。HINT 关注长视野意图保持，SA-WAM 把深度引入世界动作模型，**HRC** benchmark 和 Safe-Stop 把**接触安全**与停止可行性放到评测中心，DemoMimic 用局部接触几何提升**灵巧手**迁移，TAPVid-MV 为**多视角** 3D 跟踪补 benchmark，TTI survey 组织测试时自改进版图，PACT 则提醒我们多模型一致不等于证据可累加。对开发者来说，最有价值的是这些**开源**/**项目入口**背后的**可复现**路径：能跑、能测、能看出失败边界，才会真正推动**机器人**走向真实部署。
