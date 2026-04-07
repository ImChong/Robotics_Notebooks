# Know-How

> 本文件内容来自旧版 `README.md` 的 `Know-How` 部分，已归档至此。
> 原位置：`sources/notes/legacy-readme-resource-map.md`

---

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
