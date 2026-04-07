- Know-How
    
    - [Humanoid Robot Know-How Documentation](https://roboparty.com/roboto_origin/doc)
        
        - [ROBOTO_ORIGIN - Fully Open-Source DIY Humanoid Robot](https://github.com/Roboparty/roboto_origin?tab=readme-ov-file)
            
            - [人形机器人运动控制Know-How](https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87)
                
                - 人形机器人控制问题解决思路
                    
                    - 人形机器人与其他机器人的区别
                        
                        - 人形机器人和橡皮人
                            
                        - 人形机器人和其他机器人
                            
                        - 运动学可行和动力学可行
                            
                    - Sim2Real问题
                        
                    - 建模+求解
                        
                - 人形机器人技术框架路线展望
                    
                    - 深度强化学习运动控制方法（Learning Base)
                        
                        - BFM行为基础模型算法
                            
                            - 基于FB的无监督学习（BFM-Zero)
                                
                                - 方法局限性
                                    
                                - 基本代码
                                    
                                - 方法原理
                                    
                            - 基于Deepmimic（SONIC)
                                
                                - 方法局限性
                                    
                                - 方法原理
                                    
                            - 基于Teacher-Student多动作学习
                                
                                - 方法局限性
                                    
                                - 基本代码
                                    
                                - 方法原理
                                    
                        - AMP模仿学习+强化学习算法（仿人行走）
                            
                            - 方法局限性
                                
                            - 基本代码
                                
                            - 方法原理
                                
                        - Deepmimic模仿学习+强化学习算法（跳舞）
                            
                            - 方法局限性
                                
                            - 方法原理
                                
                        - 模仿学习基础：人-机器人重映射算法（Retarget）
                            
                            - 方法局限性
                                
                            - 方法原理
                                
                        - Attention落足点优化算法
                            
                            - 方法局限性
                                
                            - 基本代码
                                
                            - 方法原理
                                
                        - PIE感知一阶段鲁棒行走算法
                            
                            - 方法局限性
                                
                            - 基本代码
                                
                            - 方法原理
                                
                        - DreamWaq盲走一阶段鲁棒行走训练算法
                            
                            - 方法局限性
                                
                            - 基本代码
                                
                            - 方法原理
                                
                        - Teacher-Student模型和DAgger训练算法
                            
                            - 方法局限性
                                
                            - 基本代码
                                
                            - 方法原理
                                
                        - 深度强化学习理论基础（RL）
                            
                    - 传统运动控制方法（Model Base)
                        
                        - 人形机器人的状态估计
                            
                            - 方法局限性
                                
                            - 基本代码
                                
                            - 方法原理
                                
                        - 质心动力学模型+非线性模型预测控制算法+WBC（CD+NMPC+WBC）
                            
                            - 方法局限性
                                
                            - 基本代码
                                
                            - 方法原理（建模+约束+损失函数）
                                
                        - 单刚体动力学模型+凸模型预测控制算法+WBC（SRBD+Convex MPC+WBC）
                            
                            - 方法局限性
                                
                            - 基本代码
                                
                            - 方法原理（建模+约束+损失函数）
                                
                        - 全身动力学模型+全身运动控制/任务空间逆动力学优化控制算法（WBD+WBC/TSID）
                            
                            - 方法局限性
                                
                            - 基本代码
                                
                            - 方法原理（建模+约束+损失函数）
                                
                        - 弹簧负载倒立摆模型+虚拟模型控制算法（SLIP+VMC）
                            
                            - 方法局限性
                                
                            - 基本代码
                                
                            - 方法原理（建模+约束+损失函数）
                                
                        - 线性倒立摆模型+零力矩点算法（LIP+ZMP）
                            
                            - 方法局限性
                                
                            - 基本代码
                                
                            - 方法原理（建模+约束+损失函数）
                                
                        - 最优化控制问题理论基础（OCP）
                            
                - 人形机器人运动控制发展趋势
                    
                - 人形机器人运动控制学习路线
                    
                    - 强化学习运动控制学习路线
                        
                        - [【四足运控，从入门到精通】](https://www.bilibili.com/video/BV1xKabz9E2d/?spm_id_from=333.337.search-card.all.click&vd_source=30216de2308cf451749cc50ccb881d29)
                            
                    - 传统运动控制学习路线
                        
                        - 代码库
                            
                            - Pinocchio
                                
                                - 了解人形机器人浮动基动力学模型，看导入模型后的哪些变量是动力学模型里的，做几个正逆动力学的例子
                                    
                                - 了解人形机器人的正逆运动学，并通过Pinocchio+GPT做一些例子
                                    
                                - 导入人形机器人的URDF模型，看看里面包含哪些
                                    
                        - 阅读
                            
                            - 熟悉
                                
                                - 浮动基动力学模型
                                    
                                - 正逆动力学
                                    
                                - 正逆运动学
                                    
                            - 《Robot Dynamics Lecture Notes》
                                
                            - 《机器人建模和控制》
                                
    - [具身智能技术指南 Embodied-AI-Guide](https://github.com/TianxingChen/Embodied-AI-Guide/tree/main)
        
    - 深蓝学院
        
        - [人形机器人系统 - 理论与实践](https://www.shenlanxueyuan.com/course/802/task/33927/show)
            
            - 第8章: 大模型赋能人形机器人
                
            - 第7章: 人形机器人RoboCup仿真足球赛
                
            - 第6章: 基于RealSense的人形机器人感知系统
                
            - 第5章: 基于TarePlanner与FarPlanner的机器人自主探索
                
            - 第4章: 人形机器人的全局路径规划与局部避障
                
            - 第3章: 基于Lidar的人形机器人建图与定位
                
            - 第2章: 基于强化学习的人形机器人行走控制
                
            - 第1章: 人形机器人技术发展现状与课程介绍
                
        - loco-manipulation
            
            - [行走与操作：移动机器人的任务表征与全身控制](https://www.shenlanxueyuan.com/open/course/305/lesson/282/liveToVideoPreview)
                
    - AI技巧
        
        - [CS146S: The Modern Software Developer](https://themodernsoftware.dev/) [Stanford University • Fall 2025](https://themodernsoftware.dev/)
            

- Theory
    
    - Introduction to Robotics
        
        - [斯坦福《机器人学导论》](https://www.bilibili.com/video/BV17T421k78T/?vd_source=75275452d1d334b4d80721d4823e4631)
            
        - [Introduction to Robotics @ Princeton](https://www.youtube.com/playlist?list=PLF8B1bJgOQK67xkgYz_Xtx0ShjcqfdXwE)
            
        - Modern Robotics
            
            - [Video](https://www.youtube.com/playlist?list=PLggLP4f-rq02vX0OQQ5vrCxbJrzamYDfx)
                
    - Reinforcement Learning
        
        - [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) (standford)
            
        - [Reinforcement Learning:](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf) [An Introduction](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf) (CMU)
            
        - [Reinforcement Learning: An Overview](https://arxiv.org/pdf/2412.05265)
            
        - [OpenAI: Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/user/introduction.html)
            
    - [Optimal Control 2025](https://www.youtube.com/playlist?list=PLZnJoM76RM6IAJfMXd1PgGNXn3dxhkVgI)
        
        - Lecture 6: Regularization, Merit Functions, and Control History
            
        - Lecture 7: Deterministic Optimal Control and Pontryagin
            
        - Lecture 8: The Linear Quadratic Rgulator Three Ways
            
        - Lecture 9: Controllability and Dynamic Programming
            
        - Lecture 10: Convex Model-Predictive Control
            
        - Lecture 11: Nonlinear Trajectory Optimization
            
        - Lecture 12: Differential Dynamic Programming
            
        - Lecture 13: Direct Trajectory Optimization
            
        - Lecture 14: Intro to 3D Rotations
            
        - Lecture 15: Optimizing Rotations
            
        - Lecture 16: LQR with Quaternions and Quadrotors
            
        - Lecture 17: Hybrid Systems and Legged Robots
            
        - Lecture 18: Iterative Learning Control
            
        - Lecture 19: Stochastic Optimal Control and LQG
            
        - Lecture 20: How to Walk
            
        - Lecture 21: Kalman Filters and Duality
            
        - Lecture 22: Convex Relaxation and Landing Rockets
            
        - Lecture 23: Autonomous Driving and Game Theory
            
        - Lecture 24: Data-Driven Control and Behavior Cloning
            
        - Lecture 1: Intro and Dynamics Review
            
        - Lecture 2: Equilibria, Stability, and Discrete-Time Dynamics
            
        - Lecture 3~5: Optimization
            
    - [Underactuated Robotics, Spring 2024](https://www.youtube.com/playlist?list=PLkx8KyIQkMfU5szP43GlE_S1QGSPQfL9s)
        
    - [Robotic Manipulation, Fall 2023](https://www.youtube.com/playlist?list=PLkx8KyIQkMfWr191lqbN8WfV08j-ui8WX)
        
    - [Robot Dynamics 2022](https://www.youtube.com/playlist?list=PLZnJoM76RM6ItAfZIxJYNKdaR_BobleLY)
        
    - Robotics Algorithm
        
        - [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)
            
        - [CUDA Accelerated Robot Library](https://github.com/nvlabs/curobo)
            

- Paper
    
    - [Github](https://github.com/)
        
        - [Awesome-Humanoid-Robot-Learning](https://github.com/YanjieZe/awesome-humanoid-robot-learning?tab=readme-ov-file)
            
    - [arxiv](https://arxiv.org/)
        
        - [Robotics](https://arxiv.org/list/cs.RO/recent)
            

- Motion
    
    - Data Set
        
        - [AMASS](https://amass.is.tue.mpg.de/)
            
        - [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
            
        - [Mixamo](https://www.mixamo.com/#/)
            
        - [Motorica Dance Dataset](https://github.com/simonalexanderson/MotoricaDanceDataset)
            
        - [CMU Motion Capture Database](https://mocap.cs.cmu.edu/)
            
        - [NUS Mocap](https://mocap.cs.sfu.ca/nusmocap.html)
            
        - [SFU Motion Capture Database](https://mocap.cs.sfu.ca/)
            
    - Motion Generation
        
        - [HY-Motion 1.0: Scaling Flow Matching Models for Text-To-Motion Generation](https://arxiv.org/pdf/2512.23464)
            

- URDF
    
    - URDF 制作
        
        - [SolidWorks to URDF Exporter](https://wiki.ros.org/sw_urdf_exporter)
            
        - [URDFly](https://github.com/Democratizing-Dexterous/URDFly)
            
    - 开源 URDF
        
        - [Unitree](https://github.com/unitreerobotics/unitree_ros/tree/master/robots)
            
        - [Deep Robotics](https://github.com/DeepRoboticsLab/deep_robotics_model)
            
        - [MagicLabRobotics](https://github.com/MagiclabRobotics)
            
        - [awesome-humanoid-learning](https://github.com/jonyzhang2023/awesome-humanoid-learning)
            
    - URDF 可视化
        
        - [Robot Viewer](https://viewer.robotsfan.com/)
            
        - [RViz](https://docs.ros.org/en/humble/Tutorials/Intermediate/RViz/RViz-User-Guide/RViz-User-Guide.html)
            
    - URDF 处理
        
        - [Beyond URDF](https://arxiv.org/pdf/2512.23135)
            

- Retarget
    
    - [三维人体动捕模型 SMPL：A Skinned Multi Person Linear Model](https://yunyang1994.github.io/2021/08/21/%E4%B8%89%E7%BB%B4%E4%BA%BA%E4%BD%93%E6%A8%A1%E5%9E%8B-SMPL-A-Skinned-Multi-Person-Linear-Model/)
        
    - [GMR](https://github.com/YanjieZe/GMR)
        
    - [OmniRetarget](https://omniretarget.github.io/)
        
    - [SPIDER](https://github.com/facebookresearch/spider)
        
    - [PHC](https://github.com/ZhengyiLuo/PHC)
        
    - [ProtoMotions](https://github.com/NVlabs/ProtoMotions)
        
    - [GVHMR](https://github.com/zju3dv/GVHMR?tab=readme-ov-file)
        
    - [VideoMimic](https://github.com/hongsukchoi/VideoMimic)
        
    - [mocap_retarget](https://github.com/ccrpRepo/mocap_retarget?tab=readme-ov-file)
        
    - Motion Editing
        
        - [机器人关键帧编辑器](https://github.com/cyoahs/robot_motion_editor)
            
        - [robot-keyframe-kit](https://github.com/Stanford-TML/robot_keyframe_kit)
            
        - [Robot Motion Editor](https://github.com/project-instinct/robot-motion-editor)
            

- Train
    
    - Hardware
        
        - [RTX 50系显卡上PyTorch 2.4.1使用说明](https://f0exxg5fp6u.feishu.cn/wiki/IfHdw9ILaiK4kwkbitFczocKnee?from=from_copylink)
            
        - 云训练平台
            
            - [AutoDL](https://www.autodl.com/market/list)
                
            - [AWS](https://aws.amazon.com/free/webapps/?trk=3b902fe0-bcdd-4075-a945-dc3bb06c6c64&sc_channel=ps&gad_campaignid=23527793276&gbraid=0AAAAADjHtp-gQy3WZ5r1Phqzy9Hpn7131&gclid=CjwKCAiAwNDMBhBfEiwAd7ti1JIqa6Bm6_qsimEGsXJY3PoQtTJp2QGTBpfmf3txY2nt8UXMSXK57xoCmnMQAvD_BwE)
                
            - [Google Cloud](https://cloud.google.com/)
                
    - Simulator
        
        - [IsaacGym](https://docs.robotsfan.com/isaacgym/index.html)
            
        - [IsaacLab](https://isaac-sim.github.io/IsaacLab/main/index.html)
            
            - [IsaacSim](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html)
                
            - [mjlab](https://github.com/mujocolab/mjlab)
                
    - Framework
        
        - Locomotion
            
            - Reinforcement Learning
                
                - [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs)
                    
                - [Legged Gym](https://github.com/leggedrobotics/legged_gym)
                    
                - quadruped
                    
                    - [Robot Parkour Learning](https://github.com/ZiwenZhuang/parkour?tab=readme-ov-file)
                        
                - Humanoid
                    
                    - Unitree
                        
                        - [Unitree RL GYM](https://github.com/unitreerobotics/unitree_rl_gym)
                            
                        - [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab)
                            
                        - [Unitree RL Mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)
                            
                    - [Humanoid-Gym](https://github.com/roboterax/humanoid-gym)
                        
            - Imitation Learning
                
                - Whole Body Control
                    
                    - [MimicKit](https://github.com/xbpeng/MimicKit)
                        
                        - [DeepMimic](https://arxiv.org/abs/1804.02717)
                            
                    - [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking)
                        
                    - [ASAP](https://github.com/LeCAR-Lab/ASAP)
                        
                    - [PBHC-](https://kungfu-bot.github.io/)[KungfuBot](https://kungfu-bot.github.io/)
                        
                    - [OpenTrack](https://github.com/GalaxyGeneralRobotics/OpenTrack)
                        
                    - [ATOM01-Train](https://github.com/Roboparty/atom01_train)
                        
                    - [Scalable and General Whole-Body Control for Cross-Humanoid Locomotion](https://arxiv.org/html/2602.05791v1)
                        
                    - [BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning](https://github.com/LeCAR-Lab/BFM-Zero?tab=readme-ov-file)
                        
                    - [SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control](https://nvlabs.github.io/GEAR-SONIC/)
                        
                    - [OmniXtreme: Breaking the Generality Barrier in High-Dynamic Humanoid Control](https://github.com/Perkins729/OmniXtreme)
                        
                    - [MOSAIC](https://github.com/BAAI-Humanoid/MOSAIC?tab=readme-ov-file)
                        
            - Perception
                
                - [Project Instinct](https://github.com/project-instinct/InstinctLab?tab=readme-ov-file)
                    
                - [TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab)
                    
                - [FastStair: Learning to Run Up Stairs with Humanoid Robots](https://arxiv.org/pdf/2601.10365)
                    
                - [TTT-Parkour: Rapid Test-Time Training for Perceptive Robot Parkour](https://www.arxiv.org/pdf/2602.02331)
                    
                - [Hiking in the Wild: A Scalable Perceptive Parkour Framework for Humanoids](https://arxiv.org/pdf/2601.07718)
                    
                - [Real-Time Polygonal Semantic Mapping for Humanoid Robot Stair Climbing](https://arxiv.org/abs/2411.01919)
                    
        - Manipulation
            
            - Unitree
                
                - [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate)
                    
                - [kinect_teleoperate](https://github.com/unitreerobotics/kinect_teleoperate)
                    
                - [unitree_IL_lerobot](https://github.com/unitreerobotics/unitree_IL_lerobot)
                    
                - [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab)
                    
            - [RoboTwin Bimanual Robotic Manipulation Platform](https://github.com/robotwin-Platform/robotwin?tab=readme-ov-file)
                
            - [χ₀: Resource-Aware Robust Manipulation via Taming Distributional Inconsistencies](https://github.com/OpenDriveLab/kai0)
                
        - VLA
            
            - [UnifoLM-VLA-0](https://github.com/unitreerobotics/unifolm-vla)
                
            - [TextOp: Real-time Interactive Text-Driven Humanoid Robot Motion Generation and Control](https://github.com/TeleHuman/TextOp)
                
            - [HEX](https://hex-humanoid.github.io/index.html)
                
            - Benchmark
                
                - [VLA SOTA Leaderboard](https://sota.evomind-tech.com/)
                    
        - World Model
            
            - [Robotic World Model Lite](https://github.com/leggedrobotics/robotic_world_model_lite)
                
            - [LIFT: Large-scale Pretraining & Efficient Finetuning for Humanoid Control](https://github.com/bigai-ai/LIFT-humanoid)
                
            - [Self-evolving Embodied AI](https://arxiv.org/abs/2602.04411)
                
            - [UnifoLM-WMA-0: A World-Model-Action (WMA) Framework under UnifoLM Family](https://github.com/unitreerobotics/unifolm-world-model-action)
                

- Sim2Sim
    
    - Simulator
        
        - [Mujoco](https://mujoco.readthedocs.io/en/stable/overview.html)
            
            - [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground/)
                
            - Locomotion
                
                - [Unitree mujoco](https://github.com/unitreerobotics/unitree_mujoco)
                    
        - [Pybullet](https://pybullet.org/wordpress/index.php/forum-2/)
            
        - Gazebo
            
            - [Gazebo Classic](https://classic.gazebosim.org/tutorials)
                
            - [GZ](https://gazebosim.org/docs/jetty/getstarted/)
                
        - [Newton](https://newton-physics.github.io/newton/guide/overview.html)
            
    - Framework
        
        - [RoboVerse](https://github.com/RoboVerseOrg/RoboVerse)
            

- Sim2Real
    
    - 部署框架
        
        - [ROS2 Humble](https://docs.ros.org/en/humble/Installation.html)
            
        - [ros2_control](https://control.ros.org/humble/doc/getting_started/getting_started.html)
            
            - [ros2_control开发](https://rtw.b-robotized.com/master/tutorials/index.html)[模板](https://rtw.b-robotized.com/master/tutorials/index.html)
                
        - [ATOM01 ROS2 Deploy](https://github.com/Roboparty/atom01_deploy)
            
        - [WBC_Deploy Controller](https://github.com/ccrpRepo/wbc_fsm?tab=readme-ov-file)
            
        - [RoboMimic Deploy](https://github.com/ccrpRepo/RoboMimic_Deploy)
            
        - [RoboMimic Deploy - 增强版](https://github.com/Renforce-Dynamics/FSMDeployG1)
            
        - [rl_sar](https://github.com/fan-ziqi/rl_sar/tree/main)
            
    - 脚踝解算
        
        - [人形机器人闭环踝关节公式推导](https://f0exxg5fp6u.feishu.cn/docx/UuOtduWXpoxH9MxsUpxchnGpnzg?from=from_copylink)
            
        - [Closed Chain Inverse Kinematics](https://github.com/gkjohnson/closed-chain-ik-js)
            
    - 电机底软
        
        - [Unitree_SDK](https://support.unitree.com/home/zh/G1_developer)
            
    - 经验分享
        
        - [Deployment-Ready RL: Pitfalls, Lessons, and Best Practices](https://thehumanoid.ai/deployment-ready-rl-pitfalls-lessons-and-best-practices/)
            
        - [Sim2Real Actuator Gap Estimator](https://github.com/isaac-sim2real/sage?tab=readme-ov-file)
            
        - [Learning Locomotion Skills Using DeepRL: Does the Choice of Action Space Matter?](https://arxiv.org/abs/1611.01055)