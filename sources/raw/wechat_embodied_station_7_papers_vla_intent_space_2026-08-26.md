---
title: 开源论文7连发！从VLA意图蒸馏到太空机器人故障自适应，这批新作太硬核了
author: 具身智能小站
date: "2026-08-26 09:00:00"
source: "https://mp.weixin.qq.com/s/zHxwlUsj22t1oPd9Q2C-dw"
---

# 开源论文7连发！从VLA意图蒸馏到太空机器人故障自适应，这批新作太硬核了

📅 2026年8月26日

### 👋 大家好！

❝

来了！2026 年新开始的一个系列，主要是整理具身智能领域最近发表的提供开源代码或数据集的项目(论文)，希望对相关领域的小伙伴有所帮助。获取这些论文的开源项目链接，可以直接在本文中查看。欢迎转发和关注！！👇

本文汇总 7 篇近期机器人与具身智能论文，内容覆盖 VLA 行为意图蒸馏、太空机器人持续适应、张量计算、工业轻量 VLA、推进器容错、手术场景理解、事件—RGB 标定与物理反馈泛化。整体来看，这些工作共同关注一个核心问题：在数据、算力、传感与硬件都不完美的真实环境中，如何让机器人依然可部署、可适应、可验证。

**综述主线：**这一批工作的重心正在从“继续放大模型”转向“补强系统结构”：给动作解码器注入行为意图，用世界模型和特权 critic 处理故障，以 ROS 2 和小模型降低部署门槛，再用物理反馈、跨模态标定与开放代码提高真实系统的韧性。

01 · arXiv:2608.23478

🔬 **Act with Intent: Distilling Behavior Intent for Vision-Language-Action Models**

📌 **Embodied AI · VLA · Intention Distillation · Robot Manipulation**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguadKEaKI5wYa3uJn3H8Aq1ia6gu2OTKzLbqarG4tD0QZhsvctm6bQLCJwOiaYZB7v2KR40nSDbiaD7GJfia6ice9Iav6ZK15nicj4Buw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

✨ 把“动作要实现什么”蒸馏进 **VLA 解码器**，让策略不只复刻电机指令。

📖 VLA 的动作解码器主要依赖行为克隆，它能学习示范中的控制指令，却没有显式建模行为在当前指令下服务的局部目标。作者提出 **Intention Distillation（Indi）**：训练时由冻结的教师 VLM 结合当前观察、指令、粗粒度动作摘要与执行视频解释行为意图，再让部署侧 VLA 在中间解码层恢复该多模态意图表示并组织动作预测。Indi 将 GR00T-N1.7 在 SimplerEnv-Bridge 上的成功率从 **64.3% 提升至 84.7%**，真实任务平均成功率从 62.0% 提升至 68.7%。

💡 VLA 的下一步不只是预测“怎么动”，而是形成可驱动动作的**局部目的表征**。

🔗 项目链接： https://leesangoh.github.io/indi-project-page/

🔗 资料来源： https://arxiv.org/pdf/2608.23478

02 · arXiv:2608.23452

🔬 **Reward-Free Continual Adaptation for Resilient Space Robots**

📌 **Space Robotics · Continual Learning · World Model · Fault Adaptation**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgubqoxU0A3uOwMZD8b1ZW1ZWpnNGNEY7iaA9hjycpEFtRGaxRiaGhI6G6lGxUfV1XZUWibKpnf7IMEOoLl96z9j6xdP6tfbia66VkpQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

✨ 无需部署期奖励，依靠**潜状态世界模型**让太空机器人持续适应严重硬件退化。

📖 太空机器人面对轮组、推进器或执行器退化时需要在线适应，但持续强化学习通常依赖部署期奖励，而真实太空环境往往缺乏外部跟踪，难以准确计算奖励。本文提出无奖励持续学习框架：先在多样仿真中预训练潜状态世界模型，部署后冻结观察编码器与奖励预测器，仅通过无监督 rollout 更新转移动态，再完全利用更新后世界模型生成的想象轨迹训练策略。作者在行星穿越、轨道导航与精密装配三类仿真任务中验证，智能体能够在严重形态故障下恢复适应。

💡 当真实奖励不可观测时，**保留奖励结构、只校准动态模型**是一条可行的在轨学习路径。

🔗 项目链接： https://github.com/AndrejOrsula/space\_robotics\_bench

🔗 资料来源： https://arxiv.org/pdf/2608.23452

03 · arXiv:2608.23320

🔬 **ROS2SmolVLA: Enabling Small Vision-Language-Action Models for Integration into Industrial-Grade Lightweight Robots**

📌 **Industrial Robotics · Small VLA · ROS 2 · On-Premise Deployment**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUgubDczW3tIyRdlCRmvvFIo1VDmYmlfNbWicKt7BCNDy2v8QtfkhIFzu0WU6CkCgw3CmmVBWtgCUpxUSqN3IDunQXZ0mXvlAyVMpw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

✨ 把 **SmolVLA 接入 ROS 2 与 UR10e**，为工业轻量机器人提供本地部署链路。

📖 小批量、多品种生产要求机器人系统更灵活，但大模型难以在本地计算并带来合规与安全压力，许多 VLA 研究又停留在实验室硬件。ROS2SmolVLA 将 Hugging Face 的 SmolVLA 适配到 Universal Robots 轻量机器人，开源 ROS 2—SmolVLA 接口，使其能够用于工业级硬件。作者在 UR10e 上通过抓取放置任务验证功能，并给出实现指南。结果支持 SmolVLA 作为小规模、需要**本地计算**任务的可行选择。

💡 工业 VLA 的关键竞争力不只在模型指标，还在**接口、容器和真机集成成本**。

🔗 项目链接： 项目主页Docker 仓库UR10e 仿真UR10e 真机相机接口LeRobot 接口

🔗 资料来源： https://arxiv.org/pdf/2608.23320

04 · arXiv:2608.22976

🔬 **Privileged Critic Training Enables Sensor-Free Thruster Fault Adaptation in End-to-End RL**

📌 **Fault-Tolerant Control · Reinforcement Learning · Privileged Critic · Space Robotics**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguabTJfxoHJfvYwZVyCnEGZx0IqxKQoNiaDuAutIZ3lGeQibUf0SsG2SJaPrMyzcqmVtSCZvMicibP1dRo4lCcCLKYam12VutTQAzUI/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

✨ 训练时让 critic 看见真实故障，部署时 actor **无需故障传感器**也能适应推进器退化。

📖 推进器可能连续退化、完全失效或卡在常开状态，而传统故障检测依赖部署时未必具备的专用传感器。作者提出 RAFT（Recurrent Asymmetric Fault Tolerant）：训练时让 PPO 的价值函数访问真实退化状态，actor 只接收标准任务观察，并用循环记忆形成无传感器故障适应策略。在配备 8 个推进器和 1 个反作用轮的浮动平台上，面对最多 4 个同时故障，RAFT 成功率达到 **70.2%**，弥合了无故障感知基线与 oracle 策略之间 **84%** 的差距。代码、检查点和数据均已开放。

💡 特权信息不必交给部署策略；放在 **critic 端塑造训练信号**同样能留下容错能力。

🔗 项目链接： https://github.com/snt-spacer/RAFT.git

🔗 资料来源： https://arxiv.org/pdf/2608.22976

05 · arXiv:2608.22972

🔬 **Optimize Surgical Triplet Recognition: A Knowledge-Driven Mixture-of-Experts Solution**

📌 **Surgical Robotics · Video Understanding · Mixture-of-Experts · Multimodal LLM**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgubnGo8F5ib795j7GiblTf4V4ybfDSpibGuoFf59zSicNjce8TdpaKAqp37NODDrSVjtW4CoU5yVicNRicOE9X7fsia7TWg1QWtnOHsvfU/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

✨ 以知识驱动 MoE 协同识别**器械—动作—目标**，缓解手术视频中的多层优化冲突。

📖 手术三元组识别需要同时判断器械、动作、目标及其关联，是上下文感知机器人辅助手术的关键感知任务。现有方法受到组件特征纠缠、长尾类别梯度冲突以及领域知识不足三方面限制。MoeCo 通过组件定制适配器解耦时空任务特征，以协调梯度学习重平衡正负梯度，并利用知识驱动的混合专家机制动态整合多模态大模型引导的知识。CholecT45 与 CholecT50 上的实验验证了协同优化流程和动态先验融合的有效性；论文表述为代码“将开放”。

💡 医疗机器人感知不仅需要更强特征，还需要把**领域先验嵌入优化过程**。

🔗 项目链接： https://github.com/YIYIZH/MoeCo

🔗 资料来源： https://arxiv.org/pdf/2608.22972

06 · arXiv:2608.22965

🔬 **Simplified Cross-Modal Calibration for Heterogeneous Event-RGB Stereo Systems**

📌 **Event Camera · RGB Vision · Cross-Modal Calibration · Robot Perception**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgualrjwsXadOZvgWoD5e2gZcWuPuPOMGjicuYiaXtcdX9MadzUfVFa4sWWeTXlicvBGC0O3xe7j8pogrDhdL3vWz6N65t8OtDPTgqs/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

✨ 用普通显示器上的调制 ChArUco 靶标，实现**无需运动的事件—RGB 标定**。

📖 事件相机与帧相机的外参标定往往需要传感器或靶标运动、精确同步，或成本较高的事件图像重建。本文提出简单的无运动跨模态标定框架：在普通显示器上交替呈现原始与部分混合的 ChArUco 靶标，使靶标持续可被 RGB 相机观察并稳定触发事件；随后将事件粗粒度离散成帧，进行轻量去噪和内外参标定。相比最强运动式参考与静态参考，平均重投影误差分别降低 **44%** 和 **6%**，并在机器人眼在手外标定案例中保持稳定几何测量。

💡 异构感知能否规模化落地，常取决于是否有**低门槛、可重复的标定流程**。

🔗 项目链接： https://github.com/nhessenthaler/simple-evrgb-cal

🔗 资料来源： https://arxiv.org/pdf/2608.22965

07 · arXiv:2608.22701

🔬 **Physics Filtering Favors the Generalization of Robot Learning**

📌 **Physics-Informed Learning · Generalization · Feedback Control · Robot Learning**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/aGkeWWiaUguaE8pCSRhtQW4zhCfKVXQdohbgZ7USWg2G1nILCITrSokWaRDmeDD1ykPfbuiayprlkdJhV5luP6ktKBoDeR0mterAIGRABvkd8/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

✨ 用可插拔**物理过滤反馈**修正学习残差，以有限数据提升机器人跨分布泛化。

📖 机器人真实数据昂贵且采集缓慢，仅靠扩大训练集难以复制语言模型的数据规模。作者提出 PhyFilter，以物理过滤后的学习残差修正模型输出；该模块轻量、与模型无关，并通过自动学习算法优化参数，无需人工调节。PhyFilter 在四足运动、无人机机动飞行、空中操作与加速度估计四类系统上验证：四足机器人可泛化到未见地形、负载与速度，无人机能应对未见风扰，空中机械臂在风和质量不确定性下实现厘米级抓取，感知模块也能抵抗分布漂移。

💡 机器人泛化未必只能靠数据扩张，**物理结构与实时反馈**本身就是可利用的先验。

🔗 项目链接： https://github.com/JIAjindou/PhyFilter · https://scoardyy.github.io/PhyFilter

🔗 资料来源： https://arxiv.org/pdf/2608.22701

**综合观察**

这 7 篇论文呈现出一条比单纯扩大模型更务实的路线。Indi 试图回答动作“为何而做”，ROS2SmolVLA 解决小模型“怎样进工业系统”，无奖励持续适应与 RAFT 则把故障后的在线韧性前移到世界模型和训练机制中。与此同时，事件—RGB 标定与 MoeCo 分别补齐异构感知和手术语义理解，PhyFilter 用物理反馈挑战“泛化只能靠堆数据”的惯性，而张量分解工作提供更底层的计算压缩工具。对行业而言，真正有价值的开源不只是公布模型权重，而是连同接口、仿真、数据、标定工具和部署路径一起降低复现成本。
