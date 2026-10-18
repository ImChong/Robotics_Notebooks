# 机器人会写控制程序之后，能否自己造训练数据？

> 原始抓取（2026-10-18，`scripts/fetch_wechat_article.py`），仅供溯源；来源：https://mp.weixin.qq.com/s/mmnZuhiHkuzKCvdtU_XTNA

机器人会写控制程序之后，能否自己造训练数据？
Original 具身智能小站 具身智能小站
 Oct 18, 2026, 6:00 AM

阅读路线

今天更新 26 篇具身智能新论文，覆盖机器人数据生成、VLA 评测与部署、触觉与灵巧操作、人形控制与世界模型等方向。为了方便快速浏览，本期按阅读优先级分成三档。

1 篇深读2 篇跟进23 篇扫读

先深读 EmbodiedSWE 的程序求解与示范扩增，再跟进 VLAQuantBench 的闭环量化评测和 Uranus 的交互式仿真，其余按研究方向快速浏览。

先看这一篇

1让编程智能体成为机器人数据老师

EmbodiedSWE: Coding Agents for Long Horizon Dexterous Robotics

 

一句话判断：长时序操作既难采集示范，也难靠一次写对控制程序解决。EmbodiedSWE 把编程求解、轨迹验证和数据扩增连接起来，让反复调试得到的成功程序继续为 VLA 提供训练示范，适合关注数据成本与复杂任务学习的团队。

你可以带走的两点

• 先求解，再批量造数据。成功程序按场景、策略、阶段、动力学与视觉分层扩增，并经过任务评分与回放验证。六项仿真任务中，每任务示范从10条增至400条，平均成功率由18%升至69%。

• 真机证据要带上分母。用500条仿真示范微调后，台灯拆解末阶段在10次真机试验中达到20%；它说明迁移可行，也显示距离稳定完成仍有空间。

方法亮点：把编程智能体同时用作求解器与示范生成器，将单个实例的成功程序变成可验证、可扩增的监督源；基准同时记录任务结果与求解预算，避免只看展示视频。

论文：https://arxiv.org/pdf/2609.27308

代码：https://github.com/EmbodiedSWE/EmbodiedSWE

项目页：https://embodiedswe.github.io/

另外 2 篇值得跟进

2VLA 量化是否可靠，要看闭环任务

VLAQuantBench: Closed-Loop Evaluation of Post-Training Quantization for Vision-Language-Action Models

 

看点：压低数值误差并不保证机器人完成任务。VLAQuantBench 统一量化格式与作用范围，比较独立层组和联合量化，提醒部署团队同时检查成功率、校准条件与真实算子延迟。

结果：在统一 RTN 配方下，OpenVLA-OFT 的 LIBERO-Goal 成功率由 BF16 的96%降至 W8A8 的13%，每配置200轮。该结果强调格式、层范围与校准的耦合，不能推广为所有8位量化都失效；显存节省也不保证推理加速。

论文：https://arxiv.org/pdf/2609.25376

资源：https://github.com/jiuyixu25/VLAQuantBench

3用关节轨迹驱动可持续交互的视频仿真

Uranus: Building the Next-Generation Simulation Infrastructure for Embodied AI

 

看点：Uranus 接收未来关节轨迹，逐步生成多视角画面，把流式视频生成接入机器人仿真接口。它为数据与评测提供新路径，但精细接触和分布外物体状态仍有限制。

结果：实现仓库提供推理入口，另有基础权重、蒸馏权重与演示数据。论文采用 WorldOlympiad 等评测检查时序、视角与交互一致性；画面连贯不等于接触物理准确，精细操作验证仍应保留真实环境对照。

论文：https://arxiv.org/pdf/2609.24815

代码：https://github.com/D-Robotics-AI-Lab/Uranus-OSS

项目页：https://d-robotics-ai-lab.github.io/large-model-team/blog/uranus/

23 篇一眼扫完

下面每篇只保留一个判断，适合先收藏，之后按自己的研究方向回看。

04用成功与失败轨迹修正视觉奖励

HiRE: Hindsight Reward Editing for Policy Finetuning

视觉相似不等于任务做对。HiRE 用成功与失败轨迹对照，压低看似接近目标却会失败的状态奖励，给策略微调补上控制相关反馈，并在仿真和真机上比较。

论文：https://arxiv.org/pdf/2609.27068

代码：https://github.com/HiRE-Project/HiRE

项目页：https://hire-project.github.io/

05把世界模型表征蒸馏给小型策略

Think Like a World Model, Act Like a VLA: Distilling World-Model Representations into Compact Robot Policies

THAW-VLA 缓存冻结世界模型的特征，再用对齐损失训练小型策略，部署时无需教师。表征迁移不增加推理结构是主要价值；速度与显存结果限定于作者测试的硬件和配置。

论文：https://arxiv.org/pdf/2609.24682

代码：https://github.com/trungdt880/THAW-VLA

项目页：https://thaw-vla.trung-dt.com/

06把长任务计划写成可执行事件

X-Planner: Event-Structured Task Planning for Embodied Intelligence

X-Planner 用事件状态衔接指令和下游动作模型，并探索连续潜在推理接口。真机结果来自事件模式的完整系统，离线文本评分和任务进度不能直接当作成功率。

论文：https://arxiv.org/pdf/2609.25187

代码：https://github.com/X-Square-Robot/Xplanner

07人类视频有多少帮助，用统一条件比较

H2RBench: A Real-to-Sim Benchmark for Evaluating Human-to-Robot Transfer

H2RBench 重建真实场景，在统一任务、物体和机器人监督预算下比较人到机器人迁移。增加人类示范的收益因方法而异，当前覆盖刚体桌面任务，尚不能代表野外视频学习。

论文：https://arxiv.org/pdf/2609.24778

代码：https://github.com/xiaochy/H2RBench

项目页：https://h2rbench.github.io/

08预测接近目标，还要选对能成功的动作

D-JEPA: A Decision-Aligned Latent World Model

D-JEPA 用实际执行结果学习候选未来之间的关系，修正潜在距离与任务成败不一致的问题。决策对齐在匹配候选集合下评估，效果须连同动作候选生成方式理解。

论文：https://arxiv.org/pdf/2609.24749

代码：https://github.com/NEBULIS-Lab/D-JEPA

项目页：https://nebulis-lab.com/D-JEPA/

09微调行为模型，先别急着换噪声先验

The Gaussian Is Enough: Flow-Matching Priors Do Not Help When Fine-Tuning Large Behavior Models

作者在多种行为模型上比较高斯与非高斯先验，发现后者在所测微调设置中未稳定获益。编码器训练更值得优先检查；极少数据等条件仍留有例外空间。

论文：https://arxiv.org/pdf/2609.27070

项目页：https://cxu-tri.github.io/non_gaussian_FT/

10视觉鲁棒性，不能只看干净环境成功率

LIBERO-VPro: Benchmarking Closed-Loop Visual Robustness of Robotic Foundation Models

LIBERO-VPro 在执行时扰动视觉证据，覆盖缺失、延迟、多视角矛盾和场景变化。遮住物体仍然成功未必说明鲁棒，也可能依赖位置记忆，诊断实验需结合对照阅读。

论文：https://arxiv.org/pdf/2609.24350

项目页：https://huiqiongli.github.io/LIBERO-VPro/

11采集人类示范时，实时看见机器人接触

Touch2Robot: Robot Touch in the Human Demonstration Loop

Touch2Robot 把手套触觉、人体动作和物体跟踪用于实时重定向，让示范者看到目标机器人接触反馈。采集阶段纠正形态差异改善真机回放，但任务完成度不能混称成功率。

论文：https://arxiv.org/pdf/2609.24660

项目页：https://touch2robot.github.io/

12从一张图和任务指令生成仿真示范

ARSTAG: An Agentic Real2Sim2Real System for Task-Specific Robot Data Generation

ARSTAG 协调场景重建、专家生成与数据扩增，把单张图像和指令转成训练示范。围绕任务构建仿真减少逐任务工程量；真机验证覆盖抓取、放置和堆叠，仍有初始姿态限制。

论文：https://arxiv.org/pdf/2609.24563

项目页：https://boweili666.github.io/ARSTAG/

13跨手形态迁移，先保留接触结构

Dexterous Robot Manipulation from Human Demonstrations via Contact-Anchored Retargeting and Residual Policy Learning

该流程先用物理残差学习补全人类示范接触，再按接触位置和顺序重定向到机器人手，最后修复动力学可行性。四种手形态与真机双手任务展示了迁移路径，训练仍依赖仿真。

论文：https://arxiv.org/pdf/2609.24093

代码：https://github.com/DexGEM-Lab/real2sim2real

14装箱不仅看单次落点，还要看后续空间

PackLab: A Comprehensive Framework for Developing, Training, and Evaluating MLLMs in Robotic Bin Packing

PackLab 把仿真平台、专用多模态模型和分难度评测连在一起，让模型随箱内状态更新选择物体与落点。装入体积与紧凑度共同评价布局，主要证据来自物理仿真。

论文：https://arxiv.org/pdf/2609.23784

代码：https://github.com/Correr-Zhou/PackLab

15CPU 部署先算动作供给，再看反馈频率

vla.simd: Efficient CPU Inference for Language-Conditioned Manipulation

vla.simd 用 SIMD 内核与计算复用加速 CPU 推理，并把查询延迟和动作块长度一起分析。动作供给率不等于闭环反馈频率，树莓派部署数据需要连同热负载和执行方式阅读。

论文：https://arxiv.org/pdf/2609.24274

项目页：https://vla-simd.github.io/

16不用完整重建，也能闭环引导修枝

Visuomotor Robotic Pruning in Planar Orchards Using Hybrid Reinforcement Learning

这项果园修枝工作用腕部相机光流驱动闭环控制，混合离线示范与在线仿真学习。纯仿真训练后迁移实地值得关注，但范围是平面整形果树与预先指定的切点。

论文：https://arxiv.org/pdf/2609.24906

代码：https://github.com/OSUrobotics/lpy_treesim

17人难以演示的动作，先加执行护栏

Learning Beyond What Humans Can Demonstrate

GLIDE 从任务描述和遥操作代码推断失败模式，把规则插入采集与部署过程，帮助完成难以直接示范的动作。证据来自三个具体任务，应连同护栏约束一起理解策略表现。

论文：https://arxiv.org/pdf/2609.24996

项目页：https://guardrail-policy.github.io/

18语言加手势，也需要知道何时追问

MIGU: Multimodal Instruction Grounding under Uncertainty for Manipulation Planning

MIGU 融合语言语义与指向手势的三维几何不确定性，先形成目标信念，再决定执行或澄清。不确定性显式建模连接移动操作与桌面规划，当前决策只允许单轮追问。

论文：https://arxiv.org/pdf/2609.24995

项目页：https://multimodal-instruction.github.io/

19掩码做空间提示，保留完整场景

RoboMP-DINOv2: Prompts, Not Filters for Robust Robot Manipulation

RoboMP-DINOv2 在完整图像特征上加入区域提示，避免硬掩码丢掉障碍与背景线索。空间、杂乱和颜色迁移实验均在仿真中完成，颜色增强版本应单独看待。

论文：https://arxiv.org/pdf/2609.25506

代码：https://github.com/han20192019/RoboMP_DINOv2

20用手指分工兼顾灵巧与持续握力

A Quasi-Direct-Drive Underactuated Asymmetric Hand for Dexterous and Efficient Grasping and Manipulation

Berkeley QUAD Hand 让不同手指承担灵巧与握力，以非对称驱动和柔顺耦合控制体积与发热。硬件实现了 Feix 分类中29种抓型，分母为33种。

论文：https://arxiv.org/pdf/2609.27240

项目页：https://benudavis.github.io/berkeley-quadhand/

21搬运穿过杂物区，身体和物体都要避障

HOTICE: Whole-Body Humanoid Object Transportation in Cluttered Environments

HOTICE 分别建模机器人与所搬物体的避障引导，再用上下身双策略协同控制。专家到通用策略蒸馏支持复杂场景搬运，并在 G1 真机展示；不同物体形状仍有专门训练。

论文：https://arxiv.org/pdf/2609.25363

项目页：https://hotice2027.github.io/

22用上下身分组约束控制步态平滑性

Smoothness as a Constraint for Stable Humanoid Locomotion

DeCap 把上下身平滑性拆成独立物理约束，并在接近边界时加入有界屏障惩罚。相比继续调奖励权重，直接限制动作变化更便于解释，作者在真机和多地形设置中评估。

论文：https://arxiv.org/pdf/2609.24552

项目页：https://panchalutsav.github.io/decap/

23给 VLA 画出避障所需的拓扑路线

Topology-Informed Visual Prompting For Vision Language Action Policies

该方法用拓扑签名区分外观相似却需不同绕行的状态，借助仿真规划扩增示范，再让 VLM 生成视觉路径提示。训练时几何特权信息与部署时相机引导应分开理解。

论文：https://arxiv.org/pdf/2609.23944

项目页：https://topology-vla.github.io/

24用更丰富的人体动作训练人形里程计

PRIMO: Prior-Informed Odometry from Human-Motion Tracking for Humanoid Robots

PRIMO 用仿真中跟踪人体动作的数据训练本体里程计，拓宽部署控制器之外的运动分布，再加入物理与对称先验约束估计。真机比较显示改善，但各指标降幅应分别解读。

论文：https://arxiv.org/pdf/2609.23610

资源：https://github.com/Agibot-Spatial-Intelligence/PRIMO

25把单点触觉变成可供视觉网络读取的频谱

Learning tactile perception from high-bandwidth single-point sensing

SpectRobot 把高带宽单点传感信号转成时频图，让视觉编码器读取接触历史。时间与频率信息帮助识别遮挡盒内物体，实验也比较不同传感技术，尚不代表通用灵巧操作。

论文：https://arxiv.org/pdf/2609.24621

资源：https://github.com/spectrobot-project

项目页：https://huggingface.co/jogarulfop

26为视触觉仿真补上表面纹理

Norm2Tex: Augmenting Visuo-Tactile Simulations with Texture

Norm2Tex 把法线纹理转成高度信息，在触觉渲染前修改深度图，为现有模拟器增加高频表面细节。作者通过材料分类与控制任务检查纹理信息能否迁移到真机。

论文：https://arxiv.org/pdf/2609.25398

项目页：https://utn-air.github.io/norm2tex

关注本号，持续跟进具身智能开源论文前沿动态。

整理不易，欢迎点赞、转发，留言交流。

​
具身智能小站
Like
Share
Boost
Comment