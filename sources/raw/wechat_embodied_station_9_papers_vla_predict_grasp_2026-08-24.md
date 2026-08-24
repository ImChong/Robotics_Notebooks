---
title: 9篇开源具身智能论文看懂VLA、预测控制与双臂抓取
author: 具身智能小站
date: "2026-08-24 10:00:00"
source: "https://mp.weixin.qq.com/s/e0yXB8Rz4ma3CCPX8HN2CQ"
---

# 9篇开源具身智能论文看懂VLA、预测控制与双臂抓取

📅 2026年8月24日

### 👋 大家好！

❝

来了！2026 年新开始的一个系列，主要是整理具身智能领域最近发表的提供开源代码或数据集的项目(论文)，希望对相关领域的小伙伴有所帮助。获取这些论文的开源项目链接，可以直接在本文中查看。欢迎转发和关注！！👇

本文汇总 9 篇近期机器人与具身智能论文，覆盖 **VLA** 跨本体适配、反应关键型操作、层级**世界模型**、视频**世界模型**、双臂与**灵巧抓取**、平面物体操作、**HDR** 机器人视觉、低成本机械臂，以及面向无障碍的赛博物理系统。它们共同指向一个更务实的问题：机器人如何在真实世界里看得稳、反应快、受约束地规划，并以更低的数据与硬件成本完成任务。

**综述主线：**研究重心正在从“扩大模型与数据”转向“补齐部署闭环”：用自生成数据缓解本体差异，用未来预测和系统加速应对动态环境，用几何与逻辑约束提高可执行性，再以基准、传感器融合和任务工程验证真实价值。

01 · arXiv:2608.19490

🔬 **Fine-Tuning VLAs with Self-Demonstrated Generative Control for Multi-Task Manipulation**

📌 **Embodied AI · VLA · Self-Supervised Learning**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgubFWFlMjjpVoUTUh2R0RgUiaeGic87dIuOBC0icJGwYnibR5EJt8ThVbVz0oibyrDaqsCLObCTtNxgSEmaafsicBNDOYnpRAqN7ZIwYY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

✨ 让零样本 **VLA** 自己生成交互轨迹，在适配新机械臂时兼顾旧能力与新技能。

📖 现有 **VLA** 即使具备较强的语义理解与指令跟随能力，面对与预训练配置略有差异的新机器人也可能显著掉点；只用新本体的专家数据微调，又容易遗忘原有行为先验。作者提出**自监督**微调方案，将零样本 **VLA** 在线生成的交互轨迹作为额外训练数据，与专家示范共同训练。真实 ALOHA 与 RoboTwin 新基准实验表明，该方案能在目标机器人上继承原模型任务、保持通用指令跟随，并以更高**样本效率**学习新技能。

💡 跨本体适配的关键，不只是补新数据，还要显式保护基础策略原有的行为覆盖。

🔗 项目链接： https://self-supervised-control.pages.dev/

🔗 资料来源： https://arxiv.org/pdf/2608.19490v1

02 · arXiv:2608.19422

🔬 **Cyber-Physical Systems for Accessibility and Ability Augmentation: Bridging Diverse Communities**

📌 **Cyber-Physical Systems · HCI · Assistive Robotics · XR**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguaULdcvN0B2aicvvxukEJu1VwTazkal9rKo6SVZyKlrWyDa4zuY4Eo3aKkWpM7I9Tz07onnqovxiaF4nNUfVej5gWpppkuWPeDP8/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

✨ 把可穿戴设备、机器人、XR 与智能环境纳入同一能力增强研究框架。

📖 可穿戴设备、机器人、扩展现实与智能环境的融合，正在拓展赛博物理系统支持残障人士并增强感知、记忆、学习和移动能力的设计空间。论文指出，真实落地仍受制于情境感知、用户建模、自适应交互、隐私与评估等问题，并以跨 HCI、AI、机器人和无障碍研究的工作坊为载体，通过讨论、演示与协同设计提炼共性原则、技术挑战和未来方向。该文的核心贡献是社区议题与研究框架，而非新算法或性能基准。

💡 辅助机器人要从“能执行功能”走向“理解具体的人与情境”，评价体系必须同步升级。

🔗 项目链接： https://cps4all.github.io/

🔗 资料来源： https://arxiv.org/pdf/2608.19422v1

03 · arXiv:2608.19188

🔬 **PartialBiGrasp: Inferring Hidden Local Geometry for Bimanual Grasping from Partial Views**

📌 **Bimanual Manipulation · Grasping · Partial Observation**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguZd9qT03bC51CXS3Nsgia99DJb9qR6OoaC3MxCVYGJp1pqEsBrfeibBcuUPkPjPhO3OC1cCm6oq2kicnsicgib10S8xibnZiaMEBs0dOQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

✨ 从不完整点云推断被遮挡的**局部几何**，直接生成稳定、无碰撞的**双臂抓取**对。

📖 大型、重型或几何复杂物体往往只有少量可抓区域，而真实 RGB-D 观测又难以获得完整点云，使厚度、边缘与夹爪间隙等关键几何信息缺失。PartialBiGrasp 直接处理局部点云，借助卷积占据网络隐式学习抓取性、无碰撞接触区与物体厚度，再生成满足力闭合的**双臂抓取**对，并用采样优化修正不完整几何带来的歧义。解析指标、大规模仿真和真实机器人实验均显示其能在新物体的噪声局部点云上生成稳健抓取。

💡 **双臂抓取**的瓶颈正从“配对两个抓点”转向“补全与抓取相关的隐藏**局部几何**”。

🔗 项目链接： https://partialbigrasp.github.io/

🔗 资料来源： https://arxiv.org/pdf/2608.19188v1

04 · arXiv:2608.14379

🔬 **Reflex: Enabling Fast and Predictive Vision-Language-Action Models for Reaction-Critical Manipulation**

📌 **VLA · Dynamic Manipulation · Real-Time Inference**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgubGj3EBj5cBJFWRd45qKUXzTnUQt8N141niawXxcjlpKHZ5qicdLoXuxLsgn03LV4yxhyChPhEVtaXpxptP5XgibuWlyJanAWSRfU/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

✨ 把**未来预测**与**低时延**部署放进同一套 **VLA** 和动态操作评测框架。

📖 现有 **VLA** 基准多聚焦静态操作，难以衡量机器人对移动目标和时机敏感任务的真实反应能力。作者提出包含 6 个动态任务的 ReflexBench，通过解耦仿真推进与机器人控制，支持可配置时延以及同步、异步推理；并提出无需大规模机器人数据预训练的 Reflex**VLA**，以潜在**未来预测**和多帧时序融合增强预判，再用批量视觉编码与 CUDA Graph replay 降低部署时延。实验显示其持续改善动态操作表现，同时在静态基准上保持有竞争力的准确率，真实实验也验证了有效性。

💡 动态操作不能只比较策略准确率，感知到动作之间的系统时延本身就是任务变量。

🔗 项目链接： https://reflexvla.github.io/

🔗 资料来源： https://arxiv.org/pdf/2608.14379

05 · arXiv:2608.14049

🔬 **FlatLab: A Unified Methodology Framework and Simulation-Based Benchmark for Robotic Manipulation of Flat Objects**

📌 **Robotic Manipulation · Benchmark · Sim2Real**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgubl7mEHAgUTsMOfUshHsItmFPVrBnsxgB7s7bJ7RospevwIBlkCiaBjW85BxYc6oicT9GfYJpRPhsErQfevR5MM58mrgjwRIMFhU/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

✨ 统一“策略选择—动作执行”，并建立覆盖刚性与可变形平面物体的仿真基准。

📖 平放书本、木板或布料常处于难以直接夹取的构型，且材质与几何差异使启发式预操作难以泛化。作者将操作解耦为策略生成器与动作执行模块：前者从点云学习以策略为中心、对物体变化更稳健的表示，后者把长时序任务拆成可复用动作原语并动态组合轨迹。同时提出 FlatLab，提供多类刚性和可变形平面物体的高保真物理仿真、自动多模态采集及标准任务与评测协议。实验表明该方法对未见物体和类别具有更好的泛化，并优于现有基线。

💡 平面物体操作需要同时标准化“怎么测”和“怎么做”，否则泛化结论很难横向比较。

🔗 项目链接： https://flatlab-web.github.io/

🔗 资料来源： https://arxiv.org/pdf/2608.14049

06 · arXiv:2608.13678

🔬 **hint²: Hierarchical World Models for Inference-Time Temporal Logic Guidance**

📌 **World Model · Temporal Logic · Policy Steering**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgubZDEuHcQM4TcV7Xgn6nQ81wicctrtrsYSFV3Ej2YHrfDSuuAsN7hgia21T8tBcveFTTHNdaOw8YtWM3MkYBXRfEWmS4l8G0UFSY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

✨ 用高低两层**世界模型**，在**推理时**同时引导长时序任务进度与局部安全。

📖 语言条件策略通常以短动作块闭环重规划，却难以满足跨长轨迹评估的线性**时序逻辑**（**LTL**）指令。hint² 用层级**世界模型**在推理阶段引导现有短视野策略：高层模型预测动作引起的任务原子命题变化，推动 **LTL** 自动机进度；低层动力学模型预测即时状态演化，提供精确的局部安全引导。结果显示，hint² 克服了现有 **LTL** 引导**扩散**方法的局限，在 CALVIN 上优于其他**推理时**引导方法，并能完成同时包含活性与安全约束的复杂指令，真实 UR5e 实验也得到验证。

💡 长时序合规不必全部写进策略参数，也可以由分层预测在部署时持续纠偏。

🔗 项目链接： https://anonymous-hint2.github.io/

🔗 资料来源： https://arxiv.org/pdf/2608.13678

07 · arXiv:2608.13489

🔬 **DreamX-Phi 1.0: Action-Conditioned Video World Model for Robotic Manipulation**

📌 **Video World Model · Action Conditioning · Robotic Manipulation**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgubLWlrgaS7vGicQAgdIXMfd3mQCGPbGtiayt6hBvJJUibZn8QvcyrIRvxRicQ7CeVu0wRQVB3QrvibaKtHctHsq4GBYsIHUQrKzP9lY/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

✨ 同时约束机械臂轨迹、场景深度与小物体一致性，让视频预测更服从动作。

📖 动作条件视频**世界模型**即使生成画面逼真，也可能移动错误机械臂或丢失被操作物体。DreamX-Phi 1.0 根据初始观测、语言指令以及末端位姿和夹爪状态组成的动作序列预测未来画面；它以 PRoPE 风格几何编码将每条机械臂的 SE(3) 变换注入注意力，并加入轻量深度分支、SAM3 掩码和冻结的 V-JEPA 教师以维持场景几何与抓取物体一致性，再通过分布匹配蒸馏得到少步生成器。摘要报告其在 WorldArena 2.0 两条赛道分获第一与第二。

💡 机器人**世界模型**的核心指标不是“像视频”，而是对动作、几何和对象状态都忠实。

🔗 项目链接： https://github.com/AMAP-ML/DreamX-Phi

🔗 资料来源： https://arxiv.org/pdf/2608.13489

08 · arXiv:2608.16351

🔬 **Arm-Aware Guided Dexterous Grasp Generation with Arm-Agnostic Grasp Models**

📌 **Dexterous Grasping · Diffusion · Inference-Time Guidance**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUgub1UUpfdvm35fOaMibibAu4eIzdcUlicDBQicy7kFMuSWXTe68fB6z4W9A2or7hs4GlMTZzgib1FRFPnV6Otm19p7LPo8PglghVd134/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)

✨ 无需重训抓取模型，在**扩散**采样时加入机械臂与环境约束即可提升**可执行性**。

📖 只关注悬浮手姿态的**灵巧抓取**模型，难以处理避碰、工作空间边界与连续抓取；传统拒绝采样效率低，而按机械臂重训又限制泛化。该框架复用预训练的机械臂无关抓取模型，仅在**推理时**引入机械臂和环境信息，将受约束抓取写成手部位姿与机械臂构型的联合优化，并推导机械臂约束的闭式梯度。作者证明，在抓取分布由**扩散**模型表示时，该优化等价于引导式**扩散**采样。覆盖 1 万个物体、6 类场景的评测显示，其在强约束环境中生成可行抓取的概率显著提高。

💡 把整机约束作为**推理时**引导，可让同一手部生成模型迁移到不同机械臂和环境。

🔗 项目链接： https://arm-aware-dexgrasp.github.io/

🔗 资料来源： https://arxiv.org/pdf/2608.16351

09 · arXiv:2608.15968

🔬 **Tabletop Pen Manipulation With a Vision-Guided 4-DoF Arm**

📌 **Low-Cost Robotics · Computer Vision · Motion Planning**

![Image](https://mmbiz.qpic.cn/mmbiz_png/aGkeWWiaUguaqh8CeqIt06YwxJicW2rPM8feMwck8bJ3TiaLUBJmbicmuUaBzFicCvCtZpTiaLPxSkAFA8rGYl5UO963a7OjkFLrcq91WkfaCpID4/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=8)

✨ 用视觉识别与纠偏扫动补偿缺失腕部自由度，让约 200 美元机械臂完成笔具分拣。

📖 四自由度低成本机械臂缺少用于对齐夹爪与任意朝向物体的腕部旋转关节。该系统在固定俯视相机下，用 YOLO11n-OBB 定位笔具，以相机内参与 ArUco 位姿将像素坐标映射到机器人坐标，并进行颜色分类；接近固定进给方向的笔具直接抓取，角度较大的笔具则通过纠偏扫动逐步转到可抓姿态。针对 7 件笔具记录的 326 次动作中，系统完成 196 次直接抓取和 130 次纠偏扫动，可修正最高 90° 的错位，说明任务化感知与规划可部分补偿硬件自由度不足。

💡 在结构化任务里，合理利用环境和中间动作，有时比增加关节更具性价比。

🔗 项目链接： https://github.com/Anirudhpro/4DoF\_vision\_robotic\_pen\_sorting

🔗 资料来源： https://arxiv.org/pdf/2608.15968

**综合观察**

这 9 篇论文呈现出三条清晰主线。第一，**VLA** 与**世界模型**开始正面处理本体迁移、动态反应和长时序约束，能力评价也从静态成功率转向时延、预测与安全。第二，抓取研究重新把**局部几何**、机械臂可达性和环境碰撞带回生成过程，说明“手的姿态合理”不等于“整机动作可执行”。第三，FlatLab、**HDR** **多传感器**系统与四自由度机械臂表明，具身智能的进步不只来自更大的模型，也来自更好的基准、传感器配置和任务工程。需要注意的是，CPS4All 属于跨社区工作坊论文，其贡献在于议题框架与研究协同，而非算法性能。
