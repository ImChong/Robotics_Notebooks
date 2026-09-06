# 国内具身智能的开源全景，收藏这一篇就够了：76 家公司、420+ 个开源项目

Original Yuanxq Yuanxq 具身智能研究室

在小说阅读器读本章

去阅读

在公众号小说中沉浸阅读

国内具身智能开源项目清单

持续收录国内机构公开的 GitHub、Gitee 与 Hugging Face 项目，覆盖模型、数据、仿真、控制、SDK 与工具链。

知识库入口： https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main

不到两年，国内已有 76 家具身智能机构把 420+ 个开源项目放上 GitHub——从真机数据集、VLA 模型、世界模型，到机器人的训练框架、仿真环境与 SDK 驱动，覆盖了具身智能研发链的每一环。本文按五层格局把这 76 家机构的开源项目逐条列出；每条含类别标签与一句话说明，收藏这一篇，查项目不用再翻几十个 GitHub 主页。

## ▍先看五层格局

① 整机厂商与各地人形机器人创新中心（27 家）——开源“整条链路”：本体资产、RL 训练、仿真、部署 SDK 全套公开。代码是生态入口，护城河在硬件与数据。

② 具身模型 / VLA / 世界模型 / 数据公司（21 家）——开源“大脑”与“燃料”，开源是为了立标准（LeRobot 数据格式已成事实标准）、圈开发者。

③ 灵巧手与触觉公司（6 家）——小而专：手部 SDK、仿真、遥操作与触觉数据工具链，开源浓度高。

④ 大厂与底层平台（6 家）——战略卡位：阿里、腾讯、字节、小米、小鹏、地平线各有打法，多数从模型层与工具链切入。

⑤ 产业链与移动平台公司（16 家）——生态基座：机械臂、相机、雷达、控制器厂商把 SDK 与 ROS 驱动当作默认交付物。

开源最成体系的是头部整机厂，最热的是模型公司的 VLA 与世界模型，最不缺的是 SDK 驱动。

## ▍全量清单（76 家 · 424 条）

### 第一层 · 整机厂商与各地人形机器人创新中心

智元机器人（20 项）｜官方组织：https://github.com/AgibotTech、https://github.com/AimRT、https://github.com/OpenDriveLab

• Genie Sim 3.0（数据采集/工具）——文本、图像或实景重建生成USD场景，Isaac Sim与ROS 2负责物理和传感器，数据采集及五类评测共享同一任务配置。G2模型与RLinf接口使数字孪生环境可以直接进入训练。

• GE-Sim 2.0（世界模型）——当前图像、本体状态和候选动作进入生成模型，系统滚动预测未来视频与状态，独立策略服务据此完成闭环评测。它承担学习式策略试验和数据回流，物理接触精度仍由其他验证环节负责。

• AimRT（部署运行时）——C++运行时通过插件和声明式配置组织线程、资源、通信与部署，兼容ROS 2、HTTP和gRPC，并把日志、监控与性能分析作为统一基础设施；它适合承载模型服务和机器人应用模块，不直接提供运动策略。

• AgiBot-World（数据集/Benchmark）——汇集多机器人、多场景的观测、状态和动作轨迹，并提供数据下载、处理、Benchmark以及GO-1与GO-1 Air的训练、微调和评测入口；GO-1 Air移除Latent Planner，二者共享数据平台但不是同一种模型配置。

• Genie-Envisioner-V1（世界模型）——从场景和动作条件生成机器人交互视频，为策略训练、数据扩充和未来预测提供可控环境；V1仓库连接早期EnerVerse-AC与后续GE-Sim闭环世界模拟路线。

• ACoT-VLA（VLA/操作模型）——模型在动作生成前组织与任务执行相关的中间动作推理，使长时操作中的目标、状态变化和动作选择更容易关联；仓库开放模型和实验入口。

• EnerVerse-AC（世界模型）——根据机器人场景和动作条件生成未来视觉变化，用于构造操作数据和研究动作结果预测；它是Genie-Envisioner与GE-Sim-V2之前的重要历史入口。

• agibot_x1_train（运动RL/技能训练）——并行 RL 运动训练框架（含仿真任务与策略导出链路）

• agibot_x1_infer（部署运行时）——策略部署运行时：加载训练策略、下发关节命令

• agibot_x1_hardware（本体模型资产）——官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用

• AgiBotWorldChallengeICRA2026-WorldModelBaseline（世界模型）——世界模型/预测模型：按条件生成未来状态，服务训练与评测

• EWMBench（评测）——评测基准与工具

• agibot_x2_urdf（本体模型资产）——官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用

• agibot_D1_Edu-Ultra（SDK/驱动）——官方 SDK：真机控制与状态读取的统一接入层

• agillink_omnihand_sdk（SDK/驱动）——官方 SDK：真机控制与状态读取的统一接入层

• Agibot_D1_Max（SDK/驱动）——C++ SDK通过高层运动接口、状态回调和模式切换连接D1 Max本体，仓库同时提供头文件、示例、URDF和按架构编译的动态库；系统版本与SDK兼容表决定哪些接口可以使用。

• Agibot_D1_MaxPro（SDK/驱动）——SDK同时提供四足机器人的高层运动控制和底层电机或关节控制，并开放图传、点云、里程计、IMU与状态接口，使导航、感知和自定义控制能够接入D1 MaxPro。

• A3-A3U-robot-model（本体模型资产）——提供A3与A3 Ultra多个版本的URDF、MJCF、网格、RViz配置及部分SRDF，使机器人几何、关节和规划语义能够进入ROS、MuJoCo与MoveIt工具链。

• A3_Ultra_usd（本体模型资产）——仿真级USD按Isaac Sim Asset Structure 3.0组织A3 Ultra的层级、视觉与碰撞网格、质量惯量、关节限制和驱动参数，并通过Physics Variant切换通用USD、PhysX和MuJoCo物理配置。

• aimrt_mujoco_sim（仿真环境）——MuJoCo负责机器人动力学与场景执行，AimRT负责模块通信和运行时组织，为控制器、状态发布和机器人应用提供低门槛的仿真联调入口。

宇树科技（16 项）｜官方组织：https://github.com/unitreerobotics

• Unitree RL Lab（运动RL/技能训练）——Unitree将Go2、H1和G1的Isaac Lab任务接到策略导出、MuJoCo验证与实机SDK。机器人配置和部署接口处在同一仓库，能直接检查训练关节命令与真机控制协议是否一致。

• Unitree RL Gym（运动RL/技能训练）——这是Unitree官方保留的Isaac Gym时代训练链路，本体配置、PPO训练、MuJoCo Sim2Sim和实机入口仍可沿代码追踪。它主要服务旧工程维护和新旧仿真栈对照，不应与Isaac Lab版本混用。

• Unitree RL Mjlab（运动RL/技能训练）——MJLab和MuJoCo组成不依赖Omniverse的轻量训练路线，现有任务同时覆盖速度行走与动作模仿。策略回放和真机接口让开发者可以用同一模型核对训练、Sim2Sim与部署差异。

• UnifoLM-WMA-0（世界模型）——视觉观测和动作条件进入世界模型预测未来状态，动作模块再把预测与任务条件转成机器人控制序列；官方仓库提供数据处理、训练、推理、权重和G1部署入口，用于检查世界预测怎样接回真实动作闭环。

• UnifoLM-VLA-0（VLA/操作模型）——LeRobot数据先转换为HDF5和RLDS，视觉语言主干与机器人状态共同生成动作块；仓库公开多数据集训练、LIBERO评测、服务端推理和G1客户端部署入口，把数据准备、后训练与真机执行连成一条链。

• unitree_sim_isaaclab（运动RL/技能训练）——官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

• unitree_mujoco（仿真环境）——官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

• unitree_sdk2（SDK/驱动）——官方 SDK：真机控制与状态读取的统一接入层

• unitree_sdk2_python（SDK/驱动）——官方 SDK：真机控制与状态读取的统一接入层

• unitree_ros2（SDK/驱动）——官方 SDK：真机控制与状态读取的统一接入层

• xr_teleoperate（遥操作与数据采集）——遥操作与数据采集：人体/设备输入映射为机器人动作并记录示范数据

• kinect_teleoperate（遥操作与数据采集）——遥操作与数据采集：人体/设备输入映射为机器人动作并记录示范数据

• unitree_lerobot（VLA/操作模型）——VLA/策略接入：模型输出动作块驱动本体执行

• UniArmL1（遥操作与数据采集）——遥操作与数据采集：人体/设备输入映射为机器人动作并记录示范数据

• unitree_model（本体模型资产）——官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用

• unitree_slam（感知/导航）——感知/定位/建图模块：为导航与控制提供环境状态

北京人形机器人创新中心（17 项）｜官方组织：https://github.com/Open-X-Humanoid

• TienKung-Lab（运动RL/技能训练）——SMPL-X动作经GMR重定向并转换成可视化与AMP专家数据，Isaac Lab以AMP和周期步态奖励训练走跑策略，MuJoCo执行Sim2Sim，官方Deploy仓库再将导出策略接入ROS 2真机控制；它把动作数据、训练、跨引擎验证和天工实机部署连成完整运动链。

• xMimic（全身动作跟踪/技能训练）——BVH动作由xGMR重定向为机器人PKL，再转换成含最大坐标信息的NPZ；Isaac Sim 5.1中以4096环境训练PPO跟踪器，随后通过MuJoCo和ROS 2检查策略，并把ONNX模型放入Deploy_Tienkung的BeyondMimic配置完成真机入口。

• Deploy_Tienkung（仿真环境）——ROS 2控制主库接收导出的强化学习策略，通过有限状态机、标准机器人接口和控制SDK驱动天工系列机器人；配套xSIM_MUJOCO提供相同消息链路的关节、IMU、地形和位置/力矩仿真，xhumanoid_sdk补充天工3.0关节与传感器示例。

• x-humanoid-training-toolchain（运动RL/技能训练）——RoboMIND HDF5中的图像、关节状态和任务描述被转换成LeRobotDataset 2.1，随后用ACT或Diffusion Policy配置完成训练、可视化和天工操作策略适配；它提供从官方数据到可训练模型的最低可复现入口。

• HEX（VLA/操作模型）——不同人形的状态先对齐到共享身体部位槽位，统一本体预测器学习跨本体协调和时序动力学，视觉语言线索再经残差门控与流匹配动作头生成手臂、手和腰动作；腿部由低层RL全身控制器执行高层命令，明确了VLA与运动控制之间的接口。

• XR-1（VLA/操作模型）——三阶段流程先学习统一视觉—运动离散表征，再在异构视觉与机器人数据上预训练，最后按具体本体微调动作策略；官方实现统一LeRobot 2.1数据加载、跨数据集训练、权重和Franka/UR/AgileX部署脚本，并给出天工2.0适配入口。

• Pelican-VLA 0.5（VLA/操作模型）——共享Qwen3-VL主干联合视觉语言理解、未来帧和动作预测，固定容量瓶颈Token把与接触相关的视觉信息送入动作通路；当前版本重点验证注意力层面的跨场景与跨本体泛化，并明确承认从表征到可靠动作仍有缺口。

• Pelican-VL（具身Agent/规划）——Pelican-VL从视觉和语言输入形成空间理解、任务推理与高层动作目标，为下层VLA、技能或运动控制模块提供计划；项目开放多尺度模型入口。

• xGMR（动作重定向）——动作重定向：人体/MoCap 动作映射为目标本体训练参考

• xSIM_MUJOCO（仿真环境）——官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

• RoboMIND-Sim（仿真环境）——官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

• RoboMIND-dataset-utils（数据集/Benchmark）——数据采集/转换工具：同步记录并可转入 LeRobot/RLDS 训练栈

• x-humanoid-vla-simulation-benchmark（VLA/操作模型）——评测基准与工具

• embodied-skill-kit（具身Agent/规划）——具身 Agent：任务规划与技能调度

• Robo-ValueRL（评测）——评测基准与工具

• xhumanoid_sdk（SDK/驱动）——官方 SDK：真机控制与状态读取的统一接入层

• Humanoid-Occupancy（工程与工具）——感知/定位/建图模块：为导航与控制提供环境状态

优必选（5 项）｜官方组织：https://github.com/UBTECH-Robot

• WalkerS2-Model（本体模型资产）——官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用

• GlobalHumanoidRobotChallenge_2026_Baseline（评测）——评测基准与工具

• GHRC_Evaluation_2026（评测）——评测基准与工具

• Walker_TienKung_URDF（本体模型资产）——官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用

• Walker_TienKung_DEX_URDF（本体模型资产）——官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用

傅利叶智能（12 项）｜官方组织：https://github.com/FFTAI

• Wiki-GRx-Pipeline（部署运行时）——以模型准备、Isaac Gym训练、MuJoCo验证和真机部署四步连接Wiki-GRx-URDF、Wiki-GRx-MJCF、Wiki-GRx-Gym、Wiki-GRx-Mujoco与Wiki-GRx-Deploy，使读者能按同一机器人型号检查资产、策略和部署接口。

• Wiki-GRx-Deploy（部署运行时）——策略部署运行时：加载训练策略、下发关节命令

• Wiki-GRx-Gym（运动RL/技能训练）——并行 RL 运动训练框架（含仿真任务与策略导出链路）

• Wiki-GRx-Webots（仿真环境）——官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

• Wiki-GRx-Models（本体模型资产）——官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用

• Wiki-GRx-Gazebo（仿真环境）——官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

• Wiki-GRx-Mujoco（仿真环境）——官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

• teleoperation（遥操作与数据采集）——遥操作与数据采集：人体/设备输入映射为机器人动作并记录示范数据

• fourier-lerobot（VLA/操作模型）——VLA/策略接入：模型输出动作块驱动本体执行

• fourier_lab（运动RL/技能训练）——并行 RL 运动训练框架（含仿真任务与策略导出链路）

• fourier-grx-client（SDK/驱动）——官方 SDK：真机控制与状态读取的统一接入层

• fourier_dexhand_sdk（SDK/驱动）——官方 SDK：真机控制与状态读取的统一接入层

众擎机器人（7 项）｜官方组织：https://github.com/engineai-robotics

• engineai_rl_lab（全身动作跟踪/技能训练）——单段参考轨迹进入Isaac Lab的PPO跟踪任务，PM01与T800配置特权Critic、失败片段采样和全身奖励。T800额外随机化执行延迟，导出的MNN策略可接配套SDK，但仓库没有VAE与扩散阶段。

• EngineAI Native SDK（SDK/驱动）——配置文件定义FSM、机器人参数和策略资源，Runner把IMU与关节反馈整理成观测，再发送PD关节命令。MuJoCo回放、远程下发、ROS 2监测和紧急回退共同构成模型导出后的部署检查链。

• EngineAI GMR（动作重定向）——动作重定向：人体/MoCap 动作映射为目标本体训练参考

• engineai_amp（运动RL/技能训练）——并行 RL 运动训练框架（含仿真任务与策略导出链路）

• engineai_humanoid（部署运行时）——策略部署运行时：加载训练策略、下发关节命令

• engineai_ros2_workspace（SDK/驱动）——官方 SDK：真机控制与状态读取的统一接入层

• engineai_robotics_description（本体模型资产）——官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用

加速进化（7 项）｜官方组织：https://github.com/BoosterRobotics

• Booster Gym（运动RL/技能训练）——T1策略从Isaac Gym并行训练，经模型导出后进入MuJoCo、Webots和真机部署；新流水线进一步把K1的Isaac Lab训练、机器人资产与部署代码分仓组织，使关节顺序、观测契约和控制周期能够沿完整链路逐项核对。

• booster_train（运动RL/技能训练）——并行 RL 运动训练框架（含仿真任务与策略导出链路）

• booster_deploy（部署运行时）——策略部署运行时：加载训练策略、下发关节命令

• booster_robotics_sdk（SDK/驱动）——官方 SDK：真机控制与状态读取的统一接入层

• booster_robotics_sdk_ros2（SDK/驱动）——官方 SDK：真机控制与状态读取的统一接入层

• booster_assets（本体模型资产）——官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用

• robocup_demo（具身Agent/规划）——具身 Agent：任务规划与技能调度

松延动力（5 项）｜官方组织：https://github.com/Noetix-Robotics

• noetix_n2_gym（运动RL/技能训练）——为N2人形机器人提供Isaac Gym训练环境、动作加载、AMP和Sim2Sim工具。

• noetix_e1_lab（运动RL/技能训练）——为E1人形机器人提供Isaac Lab强化学习训练环境和任务配置。

• noetix_sdk_n2（SDK/驱动）——提供N2人形机器人状态读取与控制接口，连接策略运行时和真实本体。

• noetix_sdk_e1（SDK/驱动）——提供E1人形机器人控制与状态接口，用于算法部署和二次开发。

• noetix_sdk_bumi（SDK/驱动）——通过DDS提供Bumi机器人的高层与低层控制示例，支持上层应用和本体联调。

星动纪元（8 项）｜官方组织：https://github.com/roboterax

• Humanoid-Gym（运动RL/技能训练）——从Isaac Gym速度跟踪训练延伸到MuJoCo Sim2Sim和XBot实机接口，观测、奖励、关节目标与随机化都能沿链路追踪。第一次做人形行走时，可用它定位问题是在策略、模型适配还是部署循环。

• models（本体模型资产）——官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用

• ros2_sdk（SDK/驱动）——官方 SDK：真机控制与状态读取的统一接入层

• video-prediction-policy（VLA/操作模型）——VLA/策略接入：模型输出动作块驱动本体执行

• teleop_client（遥操作与数据采集）——遥操作与数据采集：人体/设备输入映射为机器人动作并记录示范数据

• xbot_sdk_api（SDK/驱动）——官方 SDK：真机控制与状态读取的统一接入层

• humanoid-lab（运动RL/技能训练）——并行 RL 运动训练框架（含仿真任务与策略导出链路）

• robotera_vla（VLA/操作模型）——VLA/策略接入：模型输出动作块驱动本体执行

乐聚机器人（8 项）｜官方组织：https://github.com/LejuRobotics

• kuavo-ros-opensource（SDK/驱动）——官方 SDK：真机控制与状态读取的统一接入层

• kuavo_data_challenge（数据集/Benchmark）——数据采集/转换工具：同步记录并可转入 LeRobot/RLDS 训练栈

• LejuLab-Train（运动RL/技能训练）——并行 RL 运动训练框架（含仿真任务与策略导出链路）

• LejuLab-Deploy（运动RL/技能训练）——策略部署运行时：加载训练策略、下发关节命令

• Leju-GMR（动作重定向）——动作重定向：人体/MoCap 动作映射为目标本体训练参考

• LeTools-Learning（运动RL/技能训练）——并行 RL 运动训练框架（含仿真任务与策略导出链路）

• letools_opensource（部署运行时）——策略部署运行时：加载训练策略、下发关节命令

• roban_model_and_sim（本体模型资产）——官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

云深处科技（10 项）｜官方组织：https://github.com/DeepRoboticsLab

• DeepRobotics RL Training（运动RL/技能训练）——Isaac Lab环境分别注册Lite3和M20粗糙地形速度跟踪任务及DR02平地AMP任务，统一RSL-RL训练、回放、键盘命令和多GPU入口；策略部署明确交给配套部署仓库，不把训练环境误写成完整实机控制栈。

• Lite3 RL Deploy（仿真环境）——把训练得到的PyTorch策略转换为ONNX，并通过策略运行器、机器人接口和Idle、StandUp、RL、JointDamping状态机接入MuJoCo或PyBullet；切换编译平台与网络配置后可连接Lite3运动主机，形成仿真检查到实机执行的连续入口。

• sdk_deploy（SDK/驱动）——策略部署运行时：加载训练策略、下发关节命令

• deep-robotics-simulation（仿真环境）——官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

• deep-robotics-teleoperate（遥操作与数据采集）——遥操作与数据采集：人体/设备输入映射为机器人动作并记录示范数据

• deep-robotics-sdk2（SDK/驱动）——官方 SDK：真机控制与状态读取的统一接入层

• Lite3_SLAM（感知/导航）——感知/定位/建图模块：为导航与控制提供环境状态

• Lite3_Navigation（工程与工具）——感知/定位/建图模块：为导航与控制提供环境状态

• fast-livo2-deep-robotics（工程与工具）——感知/定位/建图模块：为导航与控制提供环境状态

• Robot_Training_Cases（运动RL/技能训练）——并行 RL 运动训练框架（含仿真任务与策略导出链路）

逐际动力（16 项）｜官方组织：https://github.com/limxdynamics、https://github.com/FluxVLA

• TRON1 RL Isaac Gym（运动RL/技能训练）——在legged_gym结构中按机器人型号注册点足、轮足和双足TRON1环境，配置观测、奖励、随机化与PPO训练并导出策略；训练结果需继续进入MuJoCo、Gazebo或ROS部署仓库检查跨引擎与实机表现。

• TRON1 RL Deploy ROS2（仿真环境）——ROS 2控制器加载训练导出的ONNX策略，通过低层SDK读取机器人状态并下发关节命令；同一工作区组合机器人描述、Gazebo仿真和可视化工具，先检查Sim2Sim，再切换硬件接口完成TRON1实机部署。

• FluxVLA Engine（VLA/操作模型）——以统一配置和标准接口连接LeRobot数据、VLA模型组装、分布式训练、仿真评测、推理优化与机器人部署；内置多种VLM或策略适配、LIBERO与RoboCasa数据入口以及真实双臂示例，适合检查同一模型怎样从数据进入真机控制。

• humanoid-description（本体模型资产）——官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用

• humanoid-mujoco-sim（仿真环境）——官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

• humanoid-rl-deploy-python（部署运行时）——策略部署运行时：加载训练策略、下发关节命令

• humanoid-rl-isaaclab（运动RL/技能训练）——并行 RL 运动训练框架（含仿真任务与策略导出链路）

• limxsdk-lowlevel（SDK/驱动）——官方 SDK：真机控制与状态读取的统一接入层

• tron1-agent（具身Agent/规划）——具身 Agent：任务规划与技能调度

• tron1-mujoco-sim（仿真环境）——官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

• tron1-ss（工程与工具）——感知/定位/建图模块：为导航与控制提供环境状态

• tron2_env（SDK/驱动）——官方 SDK：真机控制与状态读取的统一接入层

• tron2_mujoco_sim（仿真环境）——官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

• tron2_openpi（VLA/操作模型）——VLA/策略接入：模型输出动作块驱动本体执行

• tron2_rl_lab（运动RL/技能训练）——并行 RL 运动训练框架（含仿真任务与策略导出链路）

• troncamp-mani（具身Agent/规划）——移动操作（Loco-Manip）策略接入

星海图（5 项）｜官方组织：https://github.com/OpenGalaxea

• GalaxeaVLA（VLA/操作模型）——语言、视觉和机器人状态经过VLA生成移动底盘与双臂动作，用于在星海图本体上执行多步骤移动操作任务；仓库提供模型、数据或部署入口。

• GalaxeaDP（VLA/操作模型）——把相机观测、机器人状态和任务条件映射为连续动作块，用扩散策略完成双臂或移动操作；项目适合作为GalaxeaVLA之外的模仿学习基线。

• GalaxeaManipSim（仿真环境）——官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

• GalaxeaLeRobotToolkit（数据集/Benchmark）——数据采集/转换工具：同步记录并可转入 LeRobot/RLDS 训练栈

• EFMNode（部署运行时）——策略部署运行时：加载训练策略、下发关节命令

星尘智能（3 项）｜官方组织：https://github.com/Astribot-Dev

• astribot_simulation（仿真环境）——提供星尘智能机器人相关的仿真开发环境，用于加载模型并验证机器人系统接口。

• astribot_descriptions（本体模型资产）——提供星尘智能机器人在仿真和ROS工具链中使用的描述文件与模型资产。

• astribot_msgs（仿真环境）——公开星尘智能机器人软件栈使用的自定义消息定义，为仿真与上层模块通信提供接口基础。

上海人形机器人创新中心（5 项）｜官方组织：https://github.com/loongOpen

• OpenLoong-Gymloong（运动RL/技能训练）——提供青龙人形机器人在Isaac Gym中的训练环境、任务配置和策略训练入口，用于建立基础Locomotion与Sim2Sim链路。

• OpenLoong-Dyn-Control（工程与工具）——提供人形机器人全身动力学控制软件包，将状态估计、任务目标和约束转成可执行的关节控制量。

• OpenLoong-Brain（具身Agent/规划）——提供面向人形机器人的大规模技能调度框架，把任务指令连接到可执行技能和机器人接口。

• OpenLoong-Hardware（仿真环境）——公开青龙人形机器人硬件系统资料，为控制、仿真和二次开发提供本体接口基础。

• LoongMarathonNav（工程与工具）——融合RTK、惯导、视觉和激光雷达处理城市复杂环境中的长程定位、避障与轨迹跟踪。

浙江人形机器人创新中心（1 项）｜官方组织：https://github.com/ZJ-Humanoid

• zj_humanoid_sdk_ros（SDK/驱动）——官方 SDK：真机控制与状态读取的统一接入层

小米集团（4 项）｜官方组织：https://github.com/XiaomiRobotics、https://github.com/sharinka0715

• Xiaomi-Robotics-0（VLA/操作模型）——视觉、语言和本体状态经过统一模型生成机器人动作，用于建立小米机器人团队的通用操作基线；保留首版便于比较后续版本在数据、结构和任务覆盖上的变化。

• Xiaomi-Robotics-1（VLA/操作模型）——在首版基础上扩展训练数据、操作任务和泛化评测，使视觉语言模型输出更稳定的机器人动作序列；项目用于追踪同一团队模型迭代而非只看单次演示。

• Xiaomi-Robotics-U0（VLA/操作模型）——通过统一观测与动作接口吸收不同任务或本体数据，使单一模型能够在多种机器人操作任务间迁移；项目重点在统一表示和后训练接口。

• X-WAM（世界模型）——模型联合学习视频世界变化与机器人动作，在共享表征中支持跨本体操作和未来预测；项目用于检查世界模型输出怎样与动作头连接。

小鹏机器人（2 项）｜官方组织：https://github.com/xpeng-robotics

• DIAL（VLA/操作模型）——小鹏机器人官方公开的机器人学习研究项目，具体数据、模型与动作接口以仓库和论文材料为准。

• UniT（VLA/操作模型）——小鹏机器人官方公开的统一机器人学习研究项目，用于研究多模态输入与机器人任务输出的连接方式。

智身科技（7 项）｜官方组织：https://github.com/zsibot

• matrix（仿真环境）——官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

• genisom_roamerx_open（工程与工具）——感知/定位/建图模块：为导航与控制提供环境状态

• genisom_vln（具身Agent/规划）——具身 Agent：任务规划与技能调度

• genisom_robot_sdk（SDK/驱动）——官方 SDK：真机控制与状态读取的统一接入层

• MATRiX_Python_SDK（SDK/驱动）——官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

• genisom_L1_sdk（SDK/驱动）——官方 SDK：真机控制与状态读取的统一接入层

• genisom_model（本体模型资产）——官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用

鹿明机器人（5 项）｜官方组织：https://github.com/LumosRobot

• lumos_sdk（SDK/驱动）——提供鹿明机器人C++集成接口，使状态、设备与控制功能接入上层系统。

• FastUMI_Ego（数据集/Benchmark）——提供FastUMI第一视角数据采集入口，记录人类操作过程以服务机器人模仿学习数据构建。

• FastUMI_Camera（数据集/Benchmark）——提供FastUMI板载相机工具，承担第一视角视频采集与设备接入。

• FastUMI_Data_Conversion（数据集/Benchmark）——把FastUMI采集结果转换成下游训练或分析所需格式，连接原始记录与策略训练数据。

• FastUMI_Data_Platform_Web（数据集/Benchmark）——提供FastUMI Pro数据管理Web平台，用于查看、组织和管理采集任务与数据。

达妙科技（4 项）｜官方组织：https://github.com/dmBots

• motor-sdk（SDK/驱动）——提供达妙电机设备控制与通信示例，服务关节执行器调试和机器人底层接入。

• open-dog（工程与工具）——公开达妙OpenDog01四足机器人资料，连接关节电机、机械结构和控制开发。

• wheel-legged（工程与工具）——提供达妙轮足机器人控制与开发资料，用于平衡、移动和执行器系统联调。

• bipedal-robot（工程与工具）——公开双足机器人本体与控制相关资料，为执行器、结构和运动控制实验提供基础。

高擎机电（10 项）｜官方组织：https://github.com/HighTorque-Robotics

• Mini Pi Plus AMP（运动RL/技能训练）——高擎官方仓库提供Mini Pi Plus人形机器人的AMP运动训练与仿真工具：Isaac Lab GPU并行训练、AMP+PPO与镜像轨迹增强（pi_plus_amp_sym_flat，4096并行环境、50000迭代）、MuJoCo sim2sim验证、策略回放与AMP动画、TensorBoard日志及TorchScript导出，并附带三个示例权重；README明确面向研究与仿真，实机连接前需受控验证。

• Mini-Pi-Plus_BeyondMimic（全身动作跟踪/技能训练）——全身动作跟踪策略（带仿真/真机验证）

• Mini-Pi-Plus_PBHC（全身动作跟踪/技能训练）——全身动作跟踪策略（带仿真/真机验证）

• livelybot_pi_rl_baseline（运动RL/技能训练）——并行 RL 运动训练框架（含仿真任务与策略导出链路）

• hi_dynamic_control（部署运行时）——策略部署运行时：加载训练策略、下发关节命令

• sim2real（仿真环境）——策略部署运行时：加载训练策略、下发关节命令

• Panthera-HT_Main（本体模型资产）——官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用

• Panthera-HT_SDK（SDK/驱动）——官方 SDK：真机控制与状态读取的统一接入层

• Panthera-HT_ROS2（SDK/驱动）——官方 SDK：真机控制与状态读取的统一接入层

• robot_urdf（本体模型资产）——官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用

魔法原子（5 项）｜官方组织：https://github.com/MagiclabRobotics

• magiclab_rl_lab（运动RL/技能训练）——为魔法原子机器人提供基于Isaac Lab的强化学习训练环境与任务配置。

• magicbot-mimic（全身动作跟踪/技能训练）——面向MagicBot训练动作模仿与全身跟踪策略，连接参考动作、策略和机器人关节目标。

• Magiclab_GMR（动作重定向）——把人体或其他动作源重定向到MagicBot Z1本体，为Mimic训练构建目标机器人参考动作。

• magiclab_deploy（运动RL/技能训练）——提供魔法原子运动策略的运行与部署入口，将策略输出接到机器人控制接口。

• magicbot-gen1_pi0_demo（VLA/操作模型）——展示pi0类策略接入MagicBot Gen1的模型、观测与执行接口，为VLA到真机动作链路提供示例。

越疆科技（5 项）｜官方组织：https://github.com/embodied-dobot、https://github.com/Dobot-Arm

• x-trainer（数据集/Benchmark）——为X-Trainer协作机械臂提供VR和手柄遥操作适配、数据采集与GPU仿真训练入口。

• atom-locomotion-training（运动RL/技能训练）——提供ATOM人形机器人Locomotion强化学习训练入口，连接机器人模型、任务环境和策略配置。

• atom-locomotion-deploy（仿真环境）——把ATOM运动策略接入仿真或真机运行时，处理机器人状态、策略推理与控制命令输出。

• Embodylink（数据集/Benchmark）——面向越疆机器人提供数据采集、标注、模型训练管理和遥操作工具入口。

• dobot_atom_ros2（工程与工具）——为ATOM机器人提供ROS2接口，使状态、传感器与执行器进入上层控制和开发工具链。

钛虎机器人（10 项）｜官方组织：https://github.com/ti5robot

• Ti5HandROS1SDK（SDK/驱动）——钛虎五指灵巧手 ROS1 SDK

• Ti5HandROS2SDK（SDK/驱动）——钛虎五指灵巧手 ROS2 SDK

• mechanical_arm_5_0_SDK（SDK/驱动）——钛虎机械臂 5.0 控制 SDK

• mechanical_arm_5_0_python（SDK/驱动）——钛虎机械臂 5.0 Python 控制接口

• HumanoidDualArmSolver（工程与工具）——人形双臂控制接口与解算文档

• LeftArmMotionSolver（工程与工具）——人形左手七轴运动学解算器

• RightArmMotionSolver（工程与工具）——人形右手运动学解算器

• ROS2Ti5DualArmManipulation（SDK/驱动）——人形双臂操作 ROS2 控制接口

• RoboticArm-3DVisionGrab（工程与工具）——机械臂 3D 视觉抓取示例

• multiMotorTCPAPI（SDK/驱动）——多电机 TCP 控制 API

桥介数物（1 项）｜官方组织：https://github.com/bridgedp

• hunter_bipedal_control（运动控制）——开源双足运动控制框架：非线性 MPC + 全身控制（WBC），面向 EC-hunter80 双足机器人，配套 Gazebo/MuJoCo 仿真与真机部署

萝卜派对（RoboParty）（10 项）｜官方组织：https://github.com/Roboparty

• roboto_origin（硬件开源）——全开源"手搓级"人形机器人整机（萝博头原型机）：从机械结构到控制代码完整公开（GitHub 2.3k+★）

• rpo_hardware（硬件开源）——RPO 人形硬件设计文件：机械结构、PCB、BOM 与制造资产

• UFO（运动RL/技能训练）——无监督 RL 框架（Forward-Backward + TeCH）学习可提示人形行为空间，支持动作导入与真机遥操作

• roboparty_train（运动RL/技能训练）——Isaac Lab 训练工作区：RPO 运动策略（RSL-RL）、Sim2Sim 与动作重定向

• MimicLite（全身动作跟踪/技能训练）——动作模仿训练与评测集成框架（train→adapt→finetune，发布多档检查点，可导出 ONNX 接 G1 真机链）

• Party_OS（工程与工具）——RoboParty Lab 的人形机器人系统（系统软件栈）

• roboparty_deploy（部署运行时）——RPO/Roboto 的 ROS2 部署框架：硬件驱动、推理与控制

• human-humanoid-tools（工程与工具）——人体到人形（数据/工具链）开发工具集

• roboparty_teleop（遥操作与数据采集）——全身遥操作系统

• roboparty_dexhand（SDK/驱动）——灵巧手 CAN-FD 驱动与 Python 绑定

### 第二层 · 具身模型 / VLA / 世界模型 / 数据与空间智能公司

银河通用（11 项）｜官方组织：https://github.com/GalaxyGeneralRobotics、https://github.com/PKU-EPIC

• OpenTrack / Any2Track（全身动作跟踪/技能训练）——将多个动作簇教师用DAgger蒸馏成统一JAX跟踪器，并用适配器或动力学模块处理本体与环境变化；公开仓库覆盖MuJoCo训练、检查点和部署，是研究快速适应跟踪的可运行基线。

• OpenWBT（具身Agent/规划）——头显和手柄提供视角、手部目标与行走命令，上肢通过逆运动学生成手臂目标，下肢策略负责移动和姿态调整；同一部署入口覆盖MuJoCo、Isaac Sim和真实Unitree G1，使全身遥操作链能够分层检查。

• LATENT（全身动作跟踪/技能训练）——把运动参考、球与身体状态接入全身策略，使人形机器人在移动、挥拍和平衡之间形成统一闭环；项目适合分析通用运动基座怎样被高速视觉任务调用。

• Humanoid-GPT（遥操作与数据采集）——模型把动作或语言条件编码为全身参考与关节目标，使人形机器人执行多样身体技能；仓库开放推理、权重和部署入口，适合检查生成模型与低层Tracker怎样衔接。

• GraspVLA（VLA/操作模型）——视觉与语言目标经过空间理解和抓取策略生成六自由度末端动作，用于在开放物体与场景中完成抓取；项目连接视觉语言理解、抓取候选与机器人执行。

• Click-and-Traverse（运动RL/技能训练）——用户在视觉画面中指定目标点，感知与导航模块形成局部移动目标，再由全身运动策略跨越障碍并到达目标；项目连接视觉指令、地形感知和运动基座。

• UrbanVLA（VLA/操作模型）——将第一视角视觉、语言指令与机器人状态映射为移动决策，使机器人在室外或半开放城市环境中完成目标导向导航。

• GalbotSDK（SDK/驱动）——官方 SDK：真机控制与状态读取的统一接入层

• galbot-mcap2lerobot（数据集/Benchmark）——数据采集/转换工具：同步记录并可转入 LeRobot/RLDS 训练栈

• galbot_s1_description（本体模型资产）——官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用

• HumanTracker（全身动作跟踪/技能训练）——全身动作跟踪策略（带仿真/真机验证）

千寻智能（1 项）｜官方组织：https://github.com/Spirit-AI-Team

• Spirit-v1.5（VLA/操作模型）——模型根据视觉、语言和机器人状态生成操作动作，面向多任务和真实场景泛化；仓库提供Spirit-v1.5的模型与研究入口。

自变量机器人（4 项）｜官方组织：https://github.com/X-Square-Robot

• WALL-X（VLA/操作模型）——语言、视觉和机器人状态经过统一模型生成操作动作，面向多任务与跨本体执行；仓库开放训练或推理入口，构成X Square具身模型主线。

• X-Tokenizer（VLA/操作模型）——把不同机器人或任务的连续动作编码为可由序列模型处理的Token，并解码回可执行动作；它为WALL系列跨本体训练提供动作表示基础。

• WALL-WM（世界模型）——联合建模场景视频、机器人状态与动作，预测执行后的环境变化，为动作选择和策略训练提供世界表征；仓库公开模型结构与训练评测入口。

• WALL-SS（世界模型）——当前公开论文、方法说明、实验结果和项目图片，展示动作对齐、时间尺度记忆、Dream Forcing与视觉动力学奖励对齐。现阶段适合阅读方法和跟踪发布，不能作为可直接训练或部署的代码框架。

它石智航（2 项）｜官方组织：https://github.com/tars-robotics

• RTR（VLA/操作模型）——在连续潜空间学习高频动作块，并以Reuse-then-Refine处理异步推理时新旧动作块的边界，使接触操作保持连续执行。

• World In Your Hands（数据集/Benchmark）——采集者穿戴Oracle Suite在自然工作流中产生多视角视觉、手腕和手部轨迹、压力触觉及标定数据；在线与离线算法融合红外、RGB和IMU完成动作定位，质量验证后再进行原子动作、深度、掩码、指令和推理标注。仓库提供样例数据解析、可视化、WiYH到LeRobot转换和教程，使数据可进入VLM、VLA、世界模型与跨本体操作训练。

智平方（1 项）｜官方组织：https://github.com/CHEN-H01

• Fast-in-Slow（具身Agent/规划）——以慢速推理系统组织任务并由快速策略执行操作，研究长时决策与实时动作之间的双系统接口。

智在无界（15 项）｜官方组织：https://github.com/BeingBeyond

• Being-H（数据集/Benchmark）——从大规模第一视角人类操作视频学习手部动作和任务结构，再通过目标机器人数据适配到灵巧手操作；仓库连接非本体预训练、动作表示和真机后训练。

• Being-H0（VLA/操作模型）——VLA/策略接入：模型输出动作块驱动本体执行

• Being-M0（动作重定向）——动作重定向：人体/MoCap 动作映射为目标本体训练参考

• Being-M0.5（动作重定向）——动作重定向：人体/MoCap 动作映射为目标本体训练参考

• Being-VL-0.5（世界模型）——世界模型/预测模型：按条件生成未来状态，服务训练与评测

• BumbleBee（全身动作跟踪/技能训练）——全身动作跟踪策略（带仿真/真机验证）

• DemoGrasp（具身Agent/规划）——移动操作（Loco-Manip）策略接入

• DemoHLM（具身Agent/规划）——移动操作（Loco-Manip）策略接入

• FAST（全身动作跟踪/技能训练）——全身动作跟踪策略（带仿真/真机验证）

• JALA（VLA/操作模型）——VLA/策略接入：模型输出动作块驱动本体执行

• TTP（数据集/Benchmark）——数据采集/转换工具：同步记录并可转入 LeRobot/RLDS 训练栈

• UniTacHand（具身Agent/规划）——移动操作（Loco-Manip）策略接入

• RLPF（全身动作跟踪/技能训练）——全身动作跟踪策略（带仿真/真机验证）

• Rethink_VLA（VLA/操作模型）——VLA/策略接入：模型输出动作块驱动本体执行

• VIPA-VLA（VLA/操作模型）——VLA/策略接入：模型输出动作块驱动本体执行

极佳视界（4 项）｜官方组织：https://github.com/open-gigaai

• GigaWorld-0（世界模型）——把视频外观、视角和动作建模与三维高斯场景、系统辨识及规划模块连接，形成服务VLA训练的数据生成流程；已开放训练、推理和模型配置，可核查世界建模如何产出机器人可用数据。

• GigaBrain-0（VLA/操作模型）——图像、点云、文本和本体状态进入统一模型，输出结构化任务规划与运动规划；仓库用于检查GigaWorld生成数据怎样进入GigaBrain训练和机器人任务执行链路。

• GigaWorld-Policy（VLA/操作模型）——以动作和环境变化的联合表征训练机器人策略，使世界模型不仅生成未来画面，也为动作选择提供表征；适合研究GigaWorld世界生成能力怎样转成可执行控制信号。

• GigaWorld-1（世界模型）——GigaWorld-1继续研究机器人动作条件下的未来环境生成、交互一致性和世界模型评测，为具身数据生成与策略训练提供可控环境变化。

大晓机器人（1 项）｜官方组织：https://github.com/kairos-agi

• Kairos（世界模型）——Kairos以通用视频、人类行为和真机交互数据逐级训练持续世界表征，并在统一模型中预测未来视觉状态与可执行动作；仓库开放推理代码和多组模型权重，并提供RoboTwin与LIBERO评测入口。

妙动科技（1 项）｜官方组织：https://github.com/Mondo-Robotics

• DiT4DiT（VLA/操作模型）——将视频生成DiT的中间去噪特征与流匹配动作头联合训练，使视觉未来表征直接服务机器人动作预测；仓库开放训练与评测代码，模型权重另由官方数据卡提供。

生数科技（2 项）｜官方组织：https://github.com/thu-ml、https://github.com/shengshu-ai

• Motus（世界模型）——Motus在统一架构中学习视频世界变化、语言条件和机器人动作，使同一模型既表达未来环境也支持动作预测；仓库开放模型与实验入口，用于检查世界模型怎样扩展到机器人策略。

• MotuBrain（世界模型）——MotuBrain把视频、动作和语言统一建模，并面向多本体适配、长程任务和实时闭环；公开仓库主要承载技术报告、图示和发布材料，适合了解系统定位。

原力灵机（2 项）｜官方组织：https://github.com/dexmal

• OpenDM（VLA/操作模型）——DM0.5根据语言、图像和机器人状态生成动作序列，面向开放指令、长时任务、动态干扰和多本体控制；OpenDM开放基础及任务权重、训练和推理脚本、数据注册示例，以及LIBERO、RoboTwin和SO101后训练流程。

• OpenDW（世界模型）——DW0.5接收语言、图像或视频、机器人类型、状态和动作，用共享骨干及视频、动作、价值专家联合预测未来画面、动作与状态价值；仓库开放权重、推理与训练代码，并给出RoboTwin式数据格式和动作条件回放入口。

面壁智能（2 项）｜官方组织：https://github.com/OpenBMB

• DeepThinkVLA（VLA/操作模型）——模型在视觉与语言输入到动作输出之间加入与任务执行相关的推理过程，使复杂操作中的目标、状态和动作序列能够显式关联。

• MiniCPM-Robot（具身Agent/规划）——将小型多模态模型用于机器人视觉跟踪、目标理解和动作决策，并提供Jetson、ROS 2及机器人SDK集成入口；项目强调本地断网运行和工程部署。

戴盟机器人（1 项）｜官方组织：https://github.com/dmrobot-admin

• Daimon-Infinity（数据集/Benchmark）——公开包含高分辨率触觉在内的全模态机器人数据集入口，用于Physical AI的数据预训练与操作研究。

光轮智能（5 项）｜官方组织：https://github.com/LightwheelAI

• LeIsaac（数据集/Benchmark）——在Isaac Lab中接入SO-101 Leader完成遥操作、数据采集、格式转换和后续策略训练，连接示范输入与机器人学习数据。

• LW-BenchHub（评测）——基于Isaac Lab Arena统一任务、机器人和策略接口，支持大规模具身策略评测和可复现实验配置。

• Lightwheel-YCB（仿真环境）——提供刚体、关节体和柔性物体的仿真就绪YCB资产，并同时提供MJCF与USD格式。

• Lightwheel-simready-asset（本体模型资产）——提供机器人训练和交互仿真所需的开源三维数字资产。

• LW-Egosuite-DevKit（数据集/Benchmark）——转换并可视化人类第一视角MCAP数据，用于检查相机、轨迹和多模态记录后再进入训练数据管线。

蚂蚁灵波（9 项）｜官方组织：https://github.com/Robbyant

• LingBot-VLA 2.0（VLA/操作模型）——把单臂、双臂、半人形、人形与第一视角数据映射到统一状态动作向量，稀疏MoE动作专家学习共享与本体特有模式，当前与未来感知查询分别接收深度和视频教师信号；仓库开放预训练权重、训练配置、数据映射、后训练和评测入口。

• LingBot-VA（世界模型）——视频潜变量流与机器人动作流在双流Transformer中交替建模，模型既预测动作也预测动作条件下的未来画面；仓库开放权重、RoboTwin与LIBERO后训练数据、训练和推理脚本，可检查世界预测怎样与策略输出共享表示但保持独立输出。

• LingBot-World 2.0（世界模型）——因果视频模型根据初始画面、文本与交互控制持续生成世界演化，KV缓存与蒸馏版本面向实时推理，Pilot和Director两个Agent分别组织角色行为与环境事件；它提供可交互环境模拟能力，不直接输出机器人关节动作。

• LingBot-Map（评测）——连续RGB帧经过几何上下文Transformer同时估计深度、相机轨迹与点云，窗口化推理和KV缓存支持长序列；仓库包含交互可视化、离线渲染及KITTI和Oxford Spires评测流程，适合作为空间记忆或机器人地图的感知入口。

• LingBot-Depth（工程与工具）——把RGB外观与不完整或噪声深度对齐到统一潜空间，输出补全和精修后的度量深度；仓库开放推理代码、模型权重及约三百万RGB-D样本入口，可为抓取、重建和空间感知提供更稳定的几何输入。

• LingBot-Vision（评测）——以面向几何和密集预测的自监督目标训练视觉编码器，使同一主干能够为深度、三维感知与机器人视觉任务提供特征；仓库开放代码、预训练权重和评测入口，适合比较通用视觉语义与空间几何预训练的差异。

• LingBot-Video（世界模型）——稠密与MoE视频模型从文本或图像条件生成未来视频，并通过大规模视频预训练学习场景变化与运动模式；仓库开放推理代码、模型权重和提示词重写器，可作为世界动态表征或人类视频预训练研究入口。

• LingBot-VLA 1.0（VLA/操作模型）——语言、图像和机器人状态经过具身模型生成动作块，仓库开放训练、后训练、评测和部署入口；它适合追溯LingBot-VLA从首版到2.0的接口变化，而不应再作为当前版本能力的唯一依据。

• LingBot-World 1.0（世界模型）——根据文本、初始画面或动作条件生成未来视频，用视频预训练表达机器人动作后的环境变化；它构成LingBot-World 2.0之前的技术入口，可用于比较离线视频生成与后续长时交互世界模型的差异。

蚂蚁集团（1 项）｜官方组织：https://github.com/ant-research

• Open-AoE（数据集/Benchmark）——把消费级手机视频接到数据质检、相机与手部运动恢复、原子动作标注、可视化、跨本体重定向、机器人回放、模型格式转换和VLA或世界模型训练配方。

简智机器人（1 项）｜官方组织：https://github.com/genrobot-ai

• das-datakit（数据集/Benchmark）——读取MCAP中的相机、深度、触觉和设备状态数据，提供解析、可视化及MCAP到H5转换入口，把RealOmni等采集数据接到后续清洗和训练管线。

群核科技（1 项）｜官方组织：https://github.com/manycore-research

• SpatialLM（世界模型）——空间智能大模型（NeurIPS 2025）：点云→结构化室内场景理解（墙体/门窗/物体 3D 框与语义），Llama-1B 与 Qwen-0.5B 两档，训练代码与 12,328 场景合成数据公开，常被用于机器人场景理解与仿真资产生成

求之科技（7 项）｜官方组织：https://github.com/discoverse-dev、https://github.com/DISCOVER-Robotics

• DISCOVERSE（仿真环境）——3DGS+MuJoCo 高保真 Real2Sim2Real 仿真框架（IROS 2025）：真实场景重建、仿真数据、模仿学习与实机验证一条链

• DISCOVERSE-Real2Sim（仿真环境）——真实采集数据 / 3D AIGC / 既有 3D 资产（3DGS、网格等）统一转成仿真场景

• gs_playground（仿真环境）——面向视觉信息机器人学习的高吞吐照片级仿真器

• MuJoCo-LiDAR（仿真环境）——MuJoCo 高性能 LiDAR 仿真（CPU/Warp/Taichi/JAX 多后端）

• AIRBOT-Play-Hardware（本体模型资产）——AIRBOT Play 双臂硬件开源包

• AIRBOT-Play-Hardware-with-Moveit2（SDK/驱动）——AIRBOT Play 硬件 SDK（airbot_hardware_py）与 MoveIt2 工程

• SIM2REAL-2025（数据集/Benchmark）——具身 Sim2Real 竞赛（2025）仿真与实机基线（含 ACT 等）

亮源新创（1 项）｜官方组织：https://github.com/lightorigins

• LightNav-0（感知/导航）——紧凑通用具身导航模型（Qwen3-VL 基座）：指令跟随、开放词汇物体导航与视觉跟踪，支持跨机器人形态零样本迁移（配套 HuggingFace 权重）

智澄AI（1 项）｜官方组织：https://github.com/ZhiChengAIR

• Chengling-PWM（世界模型）——机器人原生 JEPA 物理世界模型：从机器人示范学习物理动力学并预测动作轨迹（配套 TR5 Pro 人形机器人）

### 第三层 · 灵巧手与触觉传感公司

灵心巧手（5 项）｜官方组织：https://github.com/linker-bot

• linkerhand-ros2-sdk（SDK/驱动）——为LinkerHand灵巧手提供ROS2控制与状态接口，支持上层操作算法接入真实硬件。

• linkerhand-python-sdk（SDK/驱动）——提供LinkerHand的Python控制接口，用于关节命令、状态读取和机器人应用集成。

• linkerhand-sim（移动操作）——提供LinkerHand仿真环境，用于抓取、控制和操作策略在真机前的验证。

• umi-dex（数据集/Benchmark）——面向灵巧手操作采集高质量示范数据，连接人类操作输入、灵巧手状态与训练数据转换。

• linkerhand-urdf（本体模型资产）——提供LinkerHand多款灵巧手URDF模型，为仿真、规划和控制器集成提供几何与关节定义。

灵巧智能（1 项）｜官方组织：https://github.com/DexRobot

• dexrobot_ecosystem（移动操作）——整合灵巧手底层控制、运动学、URDF、Isaac Sim、MuJoCo和ROS兼容层，连接真实硬件、仿真与操作算法开发。

舞肌科技（10 项）｜官方组织：https://github.com/wuji-technology

• wujihandpy（工程与工具）——以C++核心和Python绑定提供Wuji Hand设备发现、状态读取和控制接口，是连接上层算法与真实灵巧手的基础SDK。

• wujihandros2（工程与工具）——为Wuji Hand提供ROS2状态发布与实时控制接口，可将灵巧手接入遥操作、规划和机器人学习系统。

• wuji-description（本体模型资产）——提供Wuji Hand及相关设备的URDF、MJCF、MJX和USD模型资产，连接ROS、MuJoCo、JAX和Isaac仿真流程。

• mujoco-sim（仿真环境）——提供Wuji Hand的最小MuJoCo仿真入口，用于加载模型、检查关节控制和验证上层接口。

• isaaclab-sim（运动RL/技能训练）——提供Wuji Hand在Isaac Sim和Isaac Lab中的最小仿真入口，用于模型加载、接口验证和后续训练环境集成。

• wuji-retargeting（动作重定向）——把Vision Pro获得的人手追踪结果映射为Wuji Hand可执行关节目标，为遥操作和示范采集提供人手到机器人手的映射入口。

• wuji-hand-teleop（遥操作与数据采集）——通过ROS2接入多种人体输入设备并实时控制Wuji Hand和机器人手臂，为示范采集和操作闭环提供统一入口。

• wuji-mjlab（移动操作）——基于mjlab训练Wuji Hand的手内物体旋转策略，公开PPO训练和Sim2Real部署入口，用于检查灵巧手接触控制从仿真到真机的完整链路。

• wuji-openpi（VLA/操作模型）——扩展OpenPI以支持双臂与双Wuji Hand配置，连接ROS2 MCAP示范、LeRobot数据转换、pi0或pi0.5监督微调、策略服务和真机推理。

• wuji-sdk（SDK/驱动）——提供Wuji设备发现、实时数据流和记录接口，可连接数据手套与灵巧手采集链路并为遥操作或离线数据处理提供底层输入。

帕西尼感知科技（1 项）｜官方组织：https://github.com/px-DataCollection

• px_omnisharing_dataprocess_kit（数据集/Benchmark）——触觉数据产线后处理工具链（官方首个开源项目）：把 Super EID 产线原始数据（DF-1）经双手/物体位姿估计（PX Pose）加工为 DF-2/DF-2R，并转出可直接训练的 DF-3（LeRobot 格式）；配套发布 Omnisharing 采样数据（HF 与官方数据门户）

千觉机器人（8 项）｜官方组织：https://github.com/XenseRobotics-AI、https://github.com/XenseRobotics

• xensesdk（SDK/驱动）——Xense 触觉传感器 Python SDK：深度、形变、标记点跟踪与力场数据

• xensesdk-cpp（SDK/驱动）——Xense 触觉传感器 C/C++ SDK（机器人应用集成）

• TacCap-Gripper（SDK/驱动）——TacCap 多模态触觉数据采集夹爪 SDK（C++17/Python，支持主从与腕部相机）

• XGripper（SDK/驱动）——Xense 数据采集夹爪 SDK

• xense-ros（SDK/驱动）——Xense 触觉设备 ROS 集成与示例

• lerobot-xense（工程与工具）——触觉数据接入 LeRobot 生态（Xense 物理 AI 平台）

• xense-openpi（工程与工具）——触觉数据接入 OpenPI（pi0）训练与推理链路

• xense-mcap-viewer（工程与工具）——触觉数据 MCAP 可视化工具

傲意科技（11 项）｜官方组织：https://github.com/oymotion

• gForceSDKEmbedded（SDK/驱动）——gForce 肌电臂环 C/C++ 嵌入式 SDK

• gForceSDKPython（SDK/驱动）——gForce 肌电臂环 Python SDK

• EMGFilters（工程与工具）——肌电信号滤波算法库

• ros_gforce（SDK/驱动）——gForce Pro 肌电臂环 ROS 驱动

• oglove_ros2_pkg（SDK/驱动）——OGlove 手势数据手套 ROS2 包

• rohand_ros_pkg（SDK/驱动）——ROHand 仿生灵巧手 ROS 包

• rohand_ros2_pkg（SDK/驱动）——ROHand 仿生灵巧手 ROS2 包

• rohand_gen2_urdf_ros2（本体模型资产）——ROHand Gen2 灵巧手 URDF（ROS2）

• rohand_mujoco（仿真环境）——ROHand 灵巧手 MuJoCo 仿真

• roh_gen2_firmware（工程与工具）——ROHand Gen2 固件与文档

• roh_demos（工程与工具）——ROHand 灵巧手演示工程

### 第四层 · 大厂与底层平台（互联网 / 云 / 芯片与中间件）

阿里巴巴（19 项）｜官方组织：https://github.com/amap-cvlab、https://github.com/alibaba-damo-academy

• ABot-World（世界模型）——模型接收场景条件与交互输入持续滚动生成未来世界，用于建立能够被机器人策略反复交互的视觉环境；仓库开放推理入口和模型资料，可用于分析长序列生成中的一致性与误差积累。

• RynnVLA-002（VLA/操作模型）——RynnVLA-002在视觉、语言和机器人状态条件下预测动作，面向跨任务和跨本体操作；项目用于检查Rynn系列从模型结构、训练数据到策略评测的更新。

• RynnBrain（具身Agent/规划）——视觉和语言输入先形成场景与任务表示，再输出任务步骤或技能调用，为下层VLA、导航和操作策略提供高层目标；仓库用于理解Rynn体系中大脑层与动作层的接口。

• ABot-Manipulation（VLA/操作模型）——ABot-M0.5联合处理移动与操作任务，让世界表征、动作预测和评测在同一系统中连接；仓库开放推理、预训练模型和评测入口，用于检查移动操作是否真正形成统一动作接口。

• RynnEC（具身Agent/规划）——项目研究机器人怎样从多模态观测形成环境理解、任务分解和下一步决策，并把结果交给导航或操作模块执行；适合定位认知规划层与低层策略之间的接口。

• ABot-PhysWorld（世界模型）——通过视频、动作与状态联合训练模型理解接触、位移和环境变化，并提供训练和数据入口；它位于合成数据、未来预测和动作学习之间。

• RynnVLA-001（VLA/操作模型）——把语言任务、视觉观测和机器人状态映射为动作序列，构成Rynn系列早期的通用操作基线；保留该版本有助于比较002在数据、结构和长时任务上的改动。

• ABot-Navigation（数据集/Benchmark）——视觉与语言指令经过场景理解和导航策略生成移动决策，仓库提供Benchmark、评测和方法入口；它用于检验高层语言目标怎样接到底盘导航，而不是机械臂操作。

• RynnWorld-4D（世界模型）——模型联合表达三维空间结构和时间演化，用于预测机器人动作后的场景变化与对象运动；适合作为空间理解、世界生成和规划之间的研究入口。

• RynnWorld-Teleop（遥操作与数据采集）——把操作者输入、机器人观测、状态和动作对齐成示范轨迹，并研究世界模型怎样支持数据扩充或质量判断；项目连接遥操作前端与VLA训练数据。

• RynnValue（具身Agent/规划）——模型对候选动作或执行轨迹进行价值判断，为策略选择、失败筛选和后训练提供反馈信号；它解决的是动作好坏的评估，不直接生成完整机器人控制命令。

• ABot-3DWorld（世界模型）——文本/图像/多视图/视频生成可探索 3D 世界（高德 ABot 系列）

• ABot-Claw（VLA/操作模型）——统一 VLN、VLA、WAM 与视觉记忆的持续协作机器人框架

• ABot-Explorer（具身Agent/规划）——VLM 驱动、支持 3DGS 与 Habitat 的自主探索智能体

• ABot-Recon（工程与工具）——仅用视频的长时流式 3D 重建

• AstraNav-Memory（感知/导航）——视觉上下文压缩的终身具身导航（AstraNav 系列）

• AstraNav-World（世界模型）——用于前瞻控制与一致性导航的世界模型

• CE-Nav（感知/导航）——流引导强化细化的跨本体局部导航

• OmniNav（感知/导航）——统一前瞻探索与视觉语言导航框架

腾讯机器人实验室（4 项）｜官方组织：https://github.com/Tencent-Hunyuan

• HY-Embodied（VLA/操作模型）——仓库汇总HY-Embodied系列模型、数据、训练和评测入口，使读者能够从统一位置追踪VLA、世界模型和跨本体版本之间的关系。

• HY-Embodied-0.5-VLA（VLA/操作模型）——模型接收视觉、语言和机器人状态并生成动作序列，面向多任务操作和后训练；仓库提供模型与推理入口，用于分析HY具身模型的动作接口。

• RxBrain-1.0（具身Agent/规划）——视觉与语言输入形成场景理解和任务计划，再向下层操作或导航策略发出目标；项目用于区分高层认知、策略动作与低层控制三种职责。

• HY-Embodied-0.5-X（VLA/操作模型）——通过共享多模态表征和统一动作接口学习不同机器人数据，使模型能够在多个本体和任务间迁移；仓库用于检查跨本体训练、适配和评测流程。

字节跳动机器人团队（5 项）｜官方组织：https://github.com/ByteDance-Seed、https://github.com/bytedance

• VideoWorld（世界模型）——从无标注视频学习潜在动态和行为表示，为从视觉变化中获得动作先验提供上游模型案例。

• SimArt（仿真环境）——使用多模态模型把整体网格分解为可用于仿真的关节资产，服务机器人场景与交互资产构建。

• GR-1（VLA/操作模型）——大规模视频生成预训练的视觉语言机器人操作模型（GPT 式自回归，预测动作并预测未来图像），提供代码与预训练权重（Apache-2.0）

• GR-MG（VLA/操作模型）——GR-MG 机器人操作生成模型官方实现

• Chain-of-Action（VLA/操作模型）——面向机器人操作的轨迹自回归建模（动作链推理）

百度智能云（1 项）｜官方组织：https://github.com/baidu-baige

• LoongForge（数据采集/工具）——以Megatron-LM为基础统一模型组网、数据预处理、并行策略、预训练、中期训练、SFT、LoRA和权重转换，并为VLM、VLA和扩散模型提供GPU与昆仑芯XPU训练入口。

地平线（11 项）｜官方组织：https://github.com/HorizonRobotics

• HoloAgent（具身Agent/规划）——AgentOS把语言任务展开为受监控的技能图，三维空间记忆支撑检索、执行反馈和失败恢复；当前仓库已开放机器人无关ROS 2核心、导航与感知节点、HTTP/ROS桥接、Unitree和HexFellow适配及录制工具，但模型和数据分发、无硬件快速启动与HoloAgent-1仍未完成。

• HoloMotion（全身动作跟踪/技能训练）——人体模型、重定向、动作库、稀疏MoE跟踪、评测、导出和G1部署都能沿仓库目录追踪。研发者可以固定策略扩动作库，或固定数据替换模型结构，用同一评测链判断收益来自哪一侧。

• EmbodiedGen V2（仿真环境）——语言、参考图和编辑指令生成带几何、材质、碰撞与可供性标注的仿真资产，再转换到SAPIEN、Isaac和MuJoCo。插件与并行任务入口把场景生成接到策略训练，可检验资产多样性是否带来下游泛化。

• RoboOrchardCore（数据采集/工具）——用类型化配置和批量张量容器统一相机、坐标变换、关节状态与环境数据，再由动作、观测和事件管理器组织控制循环；本地或Ray远程策略沿同一接口接入，使仿真、训练和部署代码共享显式的数据形状与运行边界。

• BIP3D（工程与工具）——感知/定位/建图模块：为导航与控制提供环境状态

• RoboOrchardLab（运动RL/技能训练）——并行 RL 运动训练框架（含仿真任务与策略导出链路）

• RoboOrchardSim（仿真环境）——官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

• RoboOrchardHardware（本体模型资产）——官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用

• RoboTransfer（世界模型）——世界模型/预测模型：按条件生成未来状态，服务训练与评测

• RoboSplatter（仿真环境）——官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

• GeoFlowSlam（感知/导航）——感知/定位/建图模块：为导航与控制提供环境状态

地瓜机器人（4 项）｜官方组织：https://github.com/D-Robotics

• rdk_LeRobot_tools（SDK/驱动）——将LeRobot模型转换和部署到地瓜RDK BPU，连接机器人学习策略与边缘计算平台。

• robot_dev_config（工程与工具）——作为TogetheROS.Bot开发入口组织机器人中间件、依赖和设备配置。

• rdk_model_zoo（SDK/驱动）——提供在RDK平台部署的模型示例与转换入口，覆盖机器人感知和边缘推理任务。

• hobot_stereonet（工程与工具）——从双目图像实时估计深度，为机器人三维感知、避障和操作提供环境几何输入。

### 第五层 · 产业链与移动平台公司（机械臂 / 底盘 / 控制器 / 传感器等）

松灵机器人（14 项）｜官方组织：https://github.com/agilexrobotics

• ARIO（本体模型资产）——官方本体模型资产（URDF/MJCF/USD），供仿真、训练与部署引用

• DataEval（数据集/Benchmark）——评测基准与工具

• data_tools（数据集/Benchmark）——数据采集/转换工具：同步记录并可转入 LeRobot/RLDS 训练栈

• gr00t-agilex（VLA/操作模型）——VLA/策略接入：模型输出动作块驱动本体执行

• lerobot-agilex（VLA/操作模型）——VLA/策略接入：模型输出动作块驱动本体执行

• openpi-agilex（VLA/操作模型）——VLA/策略接入：模型输出动作块驱动本体执行

• PikaAnyArm（遥操作与数据采集）——遥操作与数据采集：人体/设备输入映射为机器人动作并记录示范数据

• pika_ros（遥操作与数据采集）——遥操作与数据采集：人体/设备输入映射为机器人动作并记录示范数据

• pika_sdk（SDK/驱动）——官方 SDK：真机控制与状态读取的统一接入层

• piper_isaac_sim（仿真环境）——官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

• aloha-agilex（具身Agent/规划）——移动操作（Loco-Manip）策略接入

• mobile_aloha_sim（仿真环境）——官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

• mobile_aloha_sim_ros2（仿真环境）——官方仿真环境与模型接入：联调、策略回放与 Sim2Sim 验证

• AgileX Robot Lab（运动RL/技能训练）——并行 RL 运动训练框架（含仿真任务与策略导出链路）

仙工智能（3 项）｜官方组织：https://github.com/seer-robotics

• wheelDog_RL（运动RL/技能训练）——仙工智能公开的第一阶段轮足机器人强化学习项目，为其轮足平台提供训练环境与策略实验入口。

• SeerTCPTest（工程与工具）——提供Robokit NetProtocol TCP API的Qt源码与测试工具，可发送控制请求并检查移动机器人响应和日志。

• SeerSdk4j（SDK/驱动）——为仙工智能移动机器人控制器提供Java TCP客户端与请求示例，便于业务系统读取状态和调用控制接口。

众为创造（3 项）｜官方组织：https://github.com/xArm-Developer

• xarm_ros2（仿真环境）——为xArm、UFACTORY 850和Lite6提供ROS2模型、硬件接口、MoveIt规划、Gazebo仿真和真机控制示例。

• xArm-Python-SDK（SDK/驱动）——提供UFACTORY系列机械臂的官方Python API，覆盖连接、状态读取、关节与笛卡尔运动及IO控制。

• uf-gym（运动RL/技能训练）——在panda-gym基础上加入UFACTORY机器人模型和到达、抓取放置等任务，用于机械臂强化学习的快速仿真实验。

大象机器人（3 项）｜官方组织：https://github.com/elephantrobotics

• mycobot_ros2（仿真环境）——为myCobot及相关机器人提供ROS2模型、驱动、MoveIt和仿真开发入口。

• pymycobot（工程与工具）——提供大象机器人多类机械臂和双臂产品的统一Python API，用于连接、状态读取、关节与末端控制及外设集成。

• mycobot_ros（仿真环境）——为myCobot系列提供ROS模型、驱动、MoveIt配置和Gazebo仿真，可用于真机控制与上层规划验证。

奥比中光（3 项）｜官方组织：https://github.com/orbbec

• OrbbecSDK_v2（SDK/驱动）——提供奥比中光RGB-D相机底层SDK，为机器人感知、标定和数据采集读取图像与深度流。

• OrbbecSDK_ROS2（SDK/驱动）——把奥比中光相机接入ROS2，发布图像、深度和相机参数供感知与机器人应用使用。

• pyorbbecsdk（SDK/驱动）——提供OrbbecSDK的Python绑定，便于快速构建深度感知、采集和标定工具。

禾赛科技（2 项）｜官方组织：https://github.com/HesaiTechnology

• HesaiLidar_SDK_2.0（SDK/驱动）——提供禾赛激光雷达数据接收、解析和点云输出SDK，为机器人定位、感知和避障提供传感器输入。

• HesaiLidar_ROS_2.0（工程与工具）——把禾赛激光雷达接入ROS与ROS2，发布点云和设备状态供机器人系统使用。

速腾聚创（2 项）｜官方组织：https://github.com/RoboSense-LiDAR

• rslidar_sdk（SDK/驱动）——提供速腾聚创激光雷达ROS与ROS2 SDK，连接雷达数据、点云与机器人感知系统。

• rs_driver（SDK/驱动）——提供跨平台雷达驱动内核，为上层ROS或自定义应用解析速腾聚创雷达数据。

梅卡曼德（2 项）｜官方组织：https://github.com/MechMindRobotics

• mecheye_ros2_interface（工程与工具）——为Mech-Eye工业3D相机提供ROS2接口，使机器人系统能够获取图像、深度和点云并接入感知与操作流程。

• mecheye_python_samples（数据采集/工具）——提供Mech-Eye API的Python示例，覆盖相机连接、图像和点云采集以及常见参数设置。

法奥意威（2 项）｜官方组织：https://github.com/FAIR-INNOVATION

• frcobot_ros2（工程与工具）——为法奥意威协作机器人提供ROS2驱动、机器人描述、控制与MoveIt集成入口。

• fairino-python-sdk（SDK/驱动）——提供法奥意威协作机器人Python控制SDK，支持状态读取、运动命令和应用程序集成。

非夕科技（5 项）｜官方组织：https://github.com/flexivrobotics

• flexiv_rdk（SDK/驱动）——提供非夕机器人C++与Python开发接口，使控制、状态、力控和应用程序接入真实机器人。

• flexiv_ros2（工程与工具）——把Flexiv RDK接入ROS2，为感知、规划和控制应用提供标准中间件接口。

• flexiv_trainer（VLA/操作模型）——面向非夕机器人组织数据、训练和Physical AI技能开发流程，把机器人接口接到策略训练与验证。

• flexiv_tdk（数据集/Benchmark）——提供非夕机器人遥操作开发接口，用于构建主从控制、示范采集和机器人学习数据链路。

• isaac_sim_ws（仿真环境）——在Isaac Sim中加载并控制非夕机器人，并可通过真实机器人使用的RDK与力矩控制器联调。

艾利特机器人（2 项）｜官方组织：https://github.com/Elite-Robots

• Elite_Robots_CS_ROS2_Driver（SDK/驱动）——为艾利特CS系列提供ROS2驱动、标定、机器人描述、控制器、MoveIt和Gazebo仿真集成。

• Elite_Robots_CS_SDK（SDK/驱动）——提供艾利特CS系列机器人C++访问与控制库，为ROS2驱动和自定义应用提供底层接口。

节卡机器人（2 项）｜官方组织：https://github.com/JAKARobotics

• jaka_ros2（工程与工具）——为JAKA协作机器人提供ROS2驱动、机器人模型与上层规划控制接口。

• JAKA_Lumi（具身Agent/规划）——公开JAKA Lumi机器人平台相关开发内容，用于机械臂、移动平台与感知任务的系统集成。

遨博智能（2 项）｜官方组织：https://github.com/AuboRobot、https://gitee.com/aubo-nxrobo

• aubo_robot（工程与工具）——为AUBO协作机器人提供ROS工业包、机器人描述、驱动、MoveIt和Gazebo集成。

• arcs_ros2（工程与工具）——面向AUBO ARCS控制系统提供ROS2硬件接口、机器人描述和轨迹控制示例。

睿尔曼智能（5 项）｜官方组织：https://github.com/RealManRobot

• RM_API2（SDK/驱动）——提供睿尔曼机器人新一代控制API，使上层应用读取状态并发送机械臂控制命令。

• ros2_rm_robot（工程与工具）——提供睿尔曼机械臂ROS2驱动与消息接口，支持规划、感知和双臂系统集成。

• rm_models（本体模型资产）——公开睿尔曼机器人模型与仿真文件，为运动规划、碰撞检查和仿真联调提供本体描述。

• hand_eye_calibration（工程与工具）——提供睿尔曼机械臂与D435相机的眼在手和眼在外标定教程及工具。

• Touch-Teleoperation-RM65（遥操作与数据采集）——为RM65机械臂提供触控式遥操作示例，连接人机输入、运动映射与机械臂执行。

诺亦腾机器人（1 项）｜官方组织：https://github.com/noitom-robotics

• AdaPT（全身动作跟踪/技能训练）——当前仓库以mjlab提供Unitree G1发球第一阶段速度自适应Tracker：用相邻参考帧插值或外推随机改变动作执行速度，在4096个并行环境中用PPO训练；支持左右手球拍、击球臂关键帧时刻、样例动作、回放和预训练检查点。论文完整AdaPT还包含回合MVAE生成器、高层速度适配规划器、发球残差Tracker、球轨迹感知与Atom P3部署，但这些模块尚未随仓库开放。

玄雅科技（12 项）｜官方组织：https://github.com/Synria-Robotics

• RoboCore（工程与工具）——Synria 统一高吞吐机器人开发库

• Alicia-D-SDK（SDK/驱动）——Alicia-D 六轴机械臂（带夹爪）Python 控制 SDK（串口、关节/夹爪/轨迹控制与状态读取）

• Alicia-D-ROS2（SDK/驱动）——Alicia-D 机械臂 ROS2 支持包

• Alicia-D-Leader-ROS（遥操作与数据采集）——Alicia-D 主从遥操作（Leader）ROS 包

• Alicia-D-VLM-Grasp（VLA/操作模型）——Alicia-D 视觉语言模型抓取示例

• Alicia-M-SDK（SDK/驱动）——Alicia-M 机械臂 SDK

• Bessica-D-SDK（SDK/驱动）——Bessica-D 机械臂 SDK

• Gloria-M-SDK（SDK/驱动）——云犀夹爪 Python SDK

• Synria-Robots-Isaaclab（仿真环境）——Synria 机器人的 Isaac Lab 训练环境接入

• VR-Teleoperation（遥操作与数据采集）——VR 遥操作系统（虚拟遥操作 SDK）

• Electronic-Skin-ML（SDK/驱动）——ML 电子皮肤 SDK

• Open-Robot-Descriptions（本体模型资产）——开源机器人模型描述文件集合

---

预览时标签不可点

Scan to Follow

Got It

Scan with Weixin to use this Mini Program

× 分析