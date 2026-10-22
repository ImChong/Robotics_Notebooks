【10.12-10.16前沿论文动态】Manipulation 33篇
Original senlanke 具身运控lab
 Oct 20, 2026, 12:31 AM
一、VLA / 通用操作策略：5篇
1. FastVLA: Task-Conditioned and Temporally Adaptive Visual Token Compression for Efficient Vision-Language-Action Models

中文标题：FastVLA：面向高效视觉语言动作模型的任务条件与时间自适应视觉 Token 压缩

机构：论文摘要页未列明。

论文链接：https://arxiv.org/abs/2610.22146

解决的问题：VLA 需要处理大量视觉 Token，而连续控制中的相邻图像通常包含高度重复的信息。视觉计算开销会增加动作生成延迟。

创新点：将任务条件与时间变化共同引入视觉 Token 压缩。根据当前任务和视觉信息的变化调整输入表示，减少不必要的视觉计算，研究如何在保持操作能力的同时提高 VLA 推理效率。论文已被 WACV 2027 接收。

2. Multi-Teacher On-Policy Distillation for Flow-Based Vision-Language-Action Models

中文标题：面向基于 Flow 的视觉语言动作模型的多教师 On-Policy 蒸馏

机构：论文摘要页未列明。

论文链接：https://arxiv.org/abs/2610.21522

解决的问题：Flow-Based VLA 在蒸馏过程中需要同时学习教师的动作生成能力。不同教师可能具有不同的任务优势，而学生闭环执行后产生的状态分布也会不断变化。

创新点：将多教师蒸馏与 On-Policy 学习结合，使学生策略能够在自身产生的状态分布上学习教师能力。研究重点是多教师知识整合与闭环蒸馏。

3. ObjDream: Guiding Vision-Language-Action Models with Object-Level Understanding and Future State Prediction

中文标题：ObjDream：通过物体级理解与未来状态预测引导视觉语言动作模型

机构：论文摘要页未列明。

论文链接：https://arxiv.org/abs/2610.20223

解决的问题：VLA 直接从视觉和语言生成动作时，对目标物体的状态变化缺少显式预测。操作过程中，物体位置和状态变化会影响后续动作。

创新点：将物体级场景理解与未来状态预测引入 VLA，使动作决策能够利用物体状态及其预期变化，而不仅依赖当前视觉观测。

4. VeriVLA: Adaptive Test-Time Verification for Flow-Based VLAs with Latent World Models

中文标题：VeriVLA：利用潜在世界模型对 Flow-Based VLA 进行自适应测试时验证

机构：论文摘要页未列明。

论文链接：https://arxiv.org/abs/2610.21985

解决的问题：Flow-Based VLA 生成的动作轨迹可能在执行过程中偏离预期。仅依赖策略输出，难以提前判断动作是否会造成任务失败。

创新点：引入潜在世界模型进行测试时动作验证，并根据当前执行情况自适应调整验证过程。研究重点是利用未来状态预测评估动作，而不是只依赖策略自身的动作生成结果。

5. Listen Before Act: Audio-Conditioned Robot Manipulation with ListenVLA

中文标题：先听再行动：基于音频条件的 ListenVLA 机器人操作

机构：论文摘要页未列明。

论文链接：https://arxiv.org/abs/2610.16313

解决的问题：机器人操作主要依赖视觉、语言和本体感知，但部分物体状态和交互结果可以通过声音反映。仅使用视觉可能遗漏这些信息。

创新点：ListenVLA 将音频作为机器人操作策略的条件输入，研究如何利用声音信息辅助机器人理解环境和生成动作。

二、WAM / World Model 与操作预测：6篇
1. T2-z: Test-Time Optimization of Environment Latents for Robotic World Models

中文标题：T2-Z：面向机器人世界模型的测试时环境潜在变量优化

机构：斯坦福大学等（作者含李飞飞、Jiajun Wu、Yejin Choi）。

论文链接：https://arxiv.org/abs/2610.22364

解决的问题：机器人预测动作后果时，不仅需要知道当前物体的位置，还需要理解摩擦、质量等无法直接观察的环境因素。相同动作在不同物理环境中可能产生不同结果。

创新点：提出 T2-Z，在世界模型中引入可学习的环境 Latent。训练时交替执行两个步骤：固定环境 Latent 优化世界模型，再冻结世界模型优化环境 Latent。测试时不更新世界模型参数，而是根据少量真实交互优化新的环境 Latent，使模型适应未知环境。在七种仿真设置和四个真实机器人任务中，物体预测误差由 15.20 降至 7.00，动作成功率由 **33.73% 提升至 72.96%**。实验表明，仅一条测试时交互轨迹就能用于估计环境 Latent。

2. Perceive, Predict, and Plan: a Unified World Action Model for Egocentric Dexterous Manipulation

中文标题：感知、预测与规划：面向第一视角灵巧操作的统一世界动作模型

机构：待核实（作者：Zirui Wu、Ping Tan、Jie Song 等）。

论文链接：https://arxiv.org/abs/2610.22363

解决的问题：灵巧操作需要同时理解三维手部运动、预测动作产生的视觉变化，并根据任务目标规划后续动作。分别训练这些模块，难以充分共享世界动态知识。

创新点：提出 P³-WAM，将三种能力统一到一个世界动作模型中。模型包含三个运行模式：Perceive 从视频重建三维手部运动与相机轨迹；Predict 根据给定动作预测未来视觉状态；Plan 根据初始图像、语言指令和可选视觉目标，联合生成动作轨迹及其视觉后果。架构采用预训练视频 Diffusion Transformer，并通过 Mixture-of-Transformers 连接视觉和三维动作表示。三种模式共享基础参数，通过双向视觉—动作注意力交换信息。实验表明，统一训练能够同时改善三维手部重建、动作条件视频预测和目标导向规划能力。

3. UniW(A)M: A Unified Mixture-of-Experts Framework for World and World-Action Models

中文标题：UniW(A)M：面向世界模型与世界动作模型的统一混合专家框架

机构：论文摘要页未列明。

论文链接：https://arxiv.org/abs/2610.18697

解决的问题：世界模型主要预测环境变化，世界动作模型还需要生成机器人动作。两类模型的训练目标不同，独立构建会增加架构与训练流程的复杂度。

创新点：研究通过统一的 Mixture-of-Experts 架构整合世界预测和动作生成能力，使不同专家承担相应的建模任务。

4. LoopWAM: Closed-Loop Generalist Policy Improvement from Deployment Rollouts

中文标题：LoopWAM：利用部署 Rollout 实现通用策略的闭环改进

机构：论文摘要页未列明。

论文链接：https://arxiv.org/abs/2610.18873

解决的问题：通用机器人策略通常通过离线数据训练，但部署后会遇到新的状态和失败情况。离线训练数据难以持续覆盖实际执行产生的状态分布。

创新点：LoopWAM 将部署 Rollout 纳入通用策略改进流程，研究利用真实执行经验持续更新策略的闭环学习机制。

5. WAVE-0: Bridging VLA Policies and WAMs with Lightweight Latent Future Prediction

中文标题：WAVE-0：通过轻量化潜在未来预测连接 VLA 与 WAM

机构：论文摘要页未列明。

论文链接：https://arxiv.org/abs/2610.18568

解决的问题：VLA 需要未来状态信息辅助决策，但完整 WAM 的视频预测计算成本较高，不利于机器人实时控制。

创新点：研究使用轻量潜在未来预测连接 VLA 和 WAM，使策略能够利用预测信息，同时降低完整视觉未来生成带来的计算开销。

6. Box-WAM: Box Trajectories as Compact State Representations for World Action Models

中文标题：Box-WAM：使用边界框轨迹作为世界动作模型的紧凑状态表示

机构：待核实（作者：Qinxi Yu、Shenlong Wang 等）。

论文链接：https://arxiv.org/abs/2610.17526

解决的问题：WAM 通常需要处理高维视觉特征，但机器人操作中的部分关键状态变化可以通过物体位置与运动轨迹表达。

创新点：将物体边界框轨迹作为紧凑状态表示，用于世界预测和动作建模。研究重点是降低世界状态表示的复杂度，并保留与操作决策相关的物体运动信息。

三、在线/离线强化学习与后训练：3篇
1. Value as Memory: Online Value-Guided Post-training for General Robot Policies

中文标题：价值即记忆：面向通用机器人策略的在线价值引导后训练

机构：待核实（作者：Cewu Lu、Chuan Wen 等）。

论文链接：https://arxiv.org/abs/2610.22368

解决的问题：通用机器人策略在部分可观测环境中容易出现动作选择歧义。同时，离线后训练无法及时利用部署过程中获得的新反馈。

创新点：提出 VaM（Value as Memory），将最近的价值预测缓存为任务进度记忆，并将其作为动作生成的条件。训练流程结合离线示范、Off-Policy 人工干预和 On-Policy 自主 Rollout，实现从离线 RL 初始化到在线自主学习的过渡。在涉及柔性物体和复杂接触的长程操作任务中，Value Caching 带来平均 29.4 个百分点的成功率提升；在线自主学习相对匹配的离线训练提高 32.8 个百分点。

2. Trust-Calibrated VLA Policy Post-Training with Evolving Imagination

中文标题：基于信任校准与演化想象的 VLA 策略后训练

机构：论文摘要页未列明。

论文链接：https://arxiv.org/abs/2610.16517

解决的问题：利用世界模型生成的未来状态训练 VLA，可以减少真实机器人交互成本，但预测结果存在误差。错误的想象数据可能使策略更新偏离真实环境。

创新点：将信任校准引入 VLA 后训练，并结合不断更新的想象过程，研究如何控制预测信息对策略学习的影响。

3. Hierarchical Residual Policy Learning for Real-World Mobile Manipulation with Sparse Human Guidance

中文标题：基于稀疏人工指导的真实移动操作分层残差策略学习

机构：待核实（作者：Zhefei Gong、Hefeifei Jiang、Zehao Wang、Jiachen Li）。

论文链接：https://arxiv.org/abs/2610.20607

解决的问题：真实移动操作同时涉及机器人移动和物体交互，直接学习完整策略需要大量交互数据与人工指导。

创新点：将分层策略与残差策略学习结合，研究如何利用少量人工指导修正移动操作行为，降低策略适应所需的数据量。论文已被 CoRL 2026 接收。

四、模仿学习、视觉运动策略与生成式动作策略：3篇
1. EgoJEPA: Beyond Retargeting for Dexterous Manipulation from Egocentric Videos

中文标题：EgoJEPA：超越动作重定向的第一视角视频灵巧操作学习

机构：待核实（作者：Meng Cao、Ian Reid、Xiaodan Liang 等）。

论文链接：https://arxiv.org/abs/2610.20904

解决的问题：人类第一视角视频包含大量灵巧操作经验，但传统方法通常需要先估计人类手部姿态，再通过人工设计的几何映射将动作重定向到机器人。不同机器人需要重新标定，手部跟踪误差也会传递到控制策略。

创新点：提出三个阶段的学习框架。第一阶段通过 JEPA 风格的视觉预测和手部运动差异学习与动作效果相关的 Latent Action；第二阶段建立潜在动作条件世界模型，联合预测未来视觉变化和潜在动作；第三阶段使用轻量 Action Adapter 将预训练表示适配到不同机器人。整个流程不依赖人工设计的人手到机器人动作映射。在三种机器人本体、五个真实操作任务中，OOD 平均成功率达到 **74%**，重定向基线为 65%；累计跨本体训练与适配时间降低 **47.2%**。

2. Interaction Latents as Actions for Contact-Rich Manipulation

中文标题：将交互潜在变量作为复杂接触操作动作

机构：论文摘要页未列明。

论文链接：https://arxiv.org/abs/2610.18855

解决的问题：复杂接触操作不仅取决于机器人关节运动，还取决于机器人与物体之间的交互状态。直接在原始动作空间中学习，可能难以充分利用接触交互的结构。

创新点：研究将交互 Latent 作为动作表示，使策略学习围绕接触交互的潜在结构展开，而不是仅使用原始关节动作表示。

3. CFG-Policy: Unified Diffusion Policy Learning Framework via Classifier-free Guidance

中文标题：CFG-Policy：基于无分类器引导的统一扩散策略学习框架

机构：论文摘要页未列明。

论文链接：https://arxiv.org/abs/2610.14936

解决的问题：Diffusion Policy 需要根据视觉、状态和任务条件生成动作。不同条件信息对动作分布的影响需要在生成过程中进行协调。

创新点：将 Classifier-Free Guidance 引入统一的扩散策略学习框架，研究通过条件引导调整机器人动作生成过程。

五、双臂操作：2篇
1. Coordi-pi: Coordination-Aware Dual-Hand Action Representation for Bimanual Dexterous Manipulation

中文标题：Coordi-pi：面向双臂灵巧操作的协调感知双手动作表示

机构：待核实（作者：Yiming Meng、Ping Luo、Yao Mu 等）。

论文链接：https://arxiv.org/abs/2610.19753

解决的问题：双臂灵巧操作要求左右手同时完成不同但相互关联的动作。若动作表示缺少双手协调关系，策略可能难以准确学习协同操作行为。

创新点：提出协调感知的双手动作表示，将双手之间的协作关系纳入动作建模。

2. Scaling Latent Motor Adaptation to Coordinated Bimanual Manipulation

中文标题：将潜在运动适应扩展到协调双臂操作

机构：待核实（作者：Seung-Bin Joo、Alessandro Albini、Perla Maiolino 等）。

论文链接：https://arxiv.org/abs/2610.14833

解决的问题：双臂协同任务需要适应不同操作条件，同时保持左右臂之间的协调。独立调整两只手的控制行为可能破坏原有协作关系。

创新点：将潜在运动适应方法扩展到双臂协调控制，研究通过 Latent Motor Representation 调整双臂运动行为。论文已被 IROS 2026 Bimanual Manipulation Workshop 接收。

六、灵巧操作：4篇
1. Dexterity from Supervised Play

中文标题：从监督式自由交互中学习灵巧操作

机构：待核实（作者：Botao He、Yiannis Aloimonos、Jitendra Malik 等）。

论文链接：https://arxiv.org/abs/2610.22340

解决的问题：灵巧操作需要大量接触丰富的真实交互数据，但直接收集任务专用示范成本高，强化学习又可能需要大量真实机器人交互。

创新点：提出 DSP（Dexterity from Supervised Play）。首先利用几分钟人类自由交互数据，通过 Behavior Cloning 训练 Play Policy；随后让机器人自主执行该策略，收集数小时真实交互数据训练世界模型。推理时，Play Policy 为 MPC 提供候选动作序列，由世界模型评估这些动作。论文还提出在学习到的世界模型内部进行 Self-Play，将规划过程蒸馏为闭环目标条件策略。在真实灵巧手实验中，相比 Hindsight Behavior Cloning，成功率提高 **69.6%**；达到与 RL 接近的成功率时，所需交互数据减少约 490 倍。

2. HiFun: a Hierarchical Framework for Efficient Functional Dexterous Manipulation Learning

中文标题：HiFun：面向高效功能性灵巧操作学习的分层框架

机构：待核实（作者：Linyi Huang、Ping Tan、Jie Song 等）。

论文链接：https://arxiv.org/abs/2610.19518

解决的问题：灵巧手完成工具使用等功能性操作时，需要同时决定操作目标和具体手指动作。直接在高维动作空间中学习完整任务，训练复杂度较高。

创新点：提出分层灵巧操作学习框架，将功能性操作任务组织为不同层次的决策与控制过程，提高技能学习效率。

3. DexStitch: Learning and Composing Dexterous Skills from Human Videos via Contact-Aware Robotic Inpainting

中文标题：DexStitch：通过接触感知机器人补全从人类视频中学习并组合灵巧技能

机构：待核实（作者：Byeonggyeol Choi、Seongmin Jung、Yongseok Lee 等）。

论文链接：https://arxiv.org/abs/2610.17789

解决的问题：从人类视频提取的灵巧操作技能需要转换为机器人可执行动作。不同技能之间的连接还需要满足接触关系和运动连续性。

创新点：研究利用接触感知的 Robotic Inpainting 补全机器人动作，并将从人类视频学习的多个灵巧技能组合起来。

4. DexNinja: Learning Robust Dexterous Cutting Policy with a Real-to-Sim-to-Real Data Engine

中文标题：DexNinja：通过真实到仿真再到真实的数据引擎学习鲁棒灵巧切割策略

机构：待核实（作者：Haozhe Lou、Hang Zhao、Luc Van Gool 等）。

论文链接：https://arxiv.org/abs/2610.17208

解决的问题：灵巧切割任务涉及工具操作、复杂接触和物体状态变化。直接采集大量真实机器人切割数据成本较高。

创新点：将 Real-to-Sim-to-Real 数据流程用于灵巧切割策略学习，通过真实数据、仿真训练和真实部署之间的循环提高策略鲁棒性。

七、柔性/可变形物体操作：2篇
1. Continuous-Time Cloth Dynamics with Physics-Informed Neural ODEs

中文标题：基于物理信息神经常微分方程的连续时间布料动力学

机构：待核实（作者：Yingdong Ru、Lipeng Zhuang、Zhuo He、Gerardo Aragon Camarasa）。

论文链接：https://arxiv.org/abs/2610.14342

解决的问题：布料具有高维、连续变化的动力学状态。离散时间预测模型难以直接描述不同时间间隔下的布料运动。

创新点：使用 Physics-Informed Neural ODE 建立连续时间布料动力学模型，将物理约束引入神经动力学学习过程。

2. Refine Where It Matters: Contact-Adaptive Meshing for Deformable Object Manipulation

中文标题：按需细化：面向可变形物体操作的接触自适应网格划分

机构：待核实（作者：Gabriel Johnson、Tom Olesch、Fei Liu）。

论文链接：https://arxiv.org/abs/2610.14142

解决的问题：可变形物体仿真需要精细网格描述局部接触和形变，但对整个物体统一使用高分辨率网格会增加计算成本。

创新点：研究接触自适应网格划分，根据操作中的接触区域调整局部网格分辨率，将计算资源集中在更重要的形变区域。

八、人形机器人上半身操作：2篇
1. KinoGen: Customizable Humanoid Loco-Manipulation References via Kinematically Grounded Video Generation

中文标题：KinoGen：通过运动学约束视频生成构建可定制人形移动操作参考轨迹

机构：待核实（作者：Qi Wu、Lulu Zhao、Kuan Fang 等）。

论文链接：https://arxiv.org/abs/2610.20552

解决的问题：人形移动操作需要同时生成身体运动、手臂操作和物体交互的参考轨迹。普通视频生成结果不一定符合机器人的运动学约束。

创新点：将运动学约束引入视频生成过程，研究从生成视频构建可定制的人形 Loco-Manipulation 参考运动。

2. FACT: A Multimodal Force-Action-Track Foundation Model for Human-to-Robot Dexterity

中文标题：FACT：面向人类到机器人灵巧能力迁移的多模态力—动作—轨迹基础模型

机构：待核实（作者：李飞飞、Jiajun Wu、Yejin Choi 等）。

论文链接：https://arxiv.org/abs/2610.19155

解决的问题：从人类交互中学习机器人灵巧操作时，仅使用视觉轨迹难以完整表达接触力和操作动作之间的关系。

创新点：FACT 将 Force、Action 和 Track 纳入统一的多模态基础模型，研究如何联合表示接触作用、动作执行和运动轨迹，为人类操作经验向机器人迁移提供更完整的信息。

九、长程任务、推理与规划：3篇
1. Mem4Act: Learning Effective Latent Memory for Long-Horizon Robotic Manipulation

中文标题：Mem4Act：面向长程机器人操作的有效潜在记忆学习

机构：待核实（作者：Yun Zhang、Zhiyu Huang、Jiaqi Ma 等）。

论文链接：https://arxiv.org/abs/2610.22323

解决的问题：长程操作需要利用较早阶段的环境信息。仅依赖当前观测或固定长度的历史窗口，可能无法保留任务后续执行所需的关键状态。

创新点：研究通过 Latent Memory 保存对长程操作有用的历史信息，使策略在动作生成过程中利用持续更新的记忆表示。

2. ForkCast: Predictive Recovery for Vision-Language-Action Policies via Counterfactual Futures

中文标题：ForkCast：通过反事实未来预测实现视觉语言动作策略的预测式恢复

机构：论文摘要页未列明。

论文链接：https://arxiv.org/abs/2610.18650

解决的问题：VLA 在长程执行过程中可能逐渐偏离任务目标。策略需要判断当前动作是否会造成失败，并决定如何恢复。

创新点：引入反事实未来预测，研究通过比较不同动作可能产生的未来结果辅助失败恢复，而不是仅在失败发生后重新执行原动作。

3. PILOT: Guiding Long-Horizon World-Model Planning for Robot Manipulation with Visual Subgoals and Policy Priors

中文标题：PILOT：利用视觉子目标与策略先验引导长程世界模型操作规划

机构：待核实（作者：Yang Liu、Hanchen Cui、Karthik Desingh）。

论文链接：https://arxiv.org/abs/2610.14335

解决的问题：世界模型能够预测未来状态，但长程操作的动作搜索空间较大。直接在世界模型中搜索完整动作序列，可能产生较高计算成本。

创新点：将视觉子目标与策略先验结合，用于引导世界模型中的长程操作规划。通过目标信息和已有策略知识缩小规划搜索范围。

十、数据集、预训练、跨本体与硬件平台：3篇
1. LIBERO-PeRM: Benchmarking Personalized Robotic Manipulation

中文标题：LIBERO-PeRM：个性化机器人操作评测基准

机构：待核实（作者：Zhixu Li、Keqian Tang、Jiachen Li 等）。

论文链接：https://arxiv.org/abs/2610.22372

解决的问题：通用机器人不仅需要完成任务，还需要遵循不同用户的偏好。例如同一个整理任务，不同用户可能要求不同的操作顺序、动作方式或物体摆放习惯。现有 Benchmark 很难区分任务成功与偏好满足。

创新点：基于 LIBERO 建立个性化操作评测体系，覆盖运动偏好、技能偏好和用户配置偏好。提出五组测试，共 312 个任务，分别评估零样本偏好遵循、从示范学习单个偏好、数据效率、跨场景泛化以及偏好组合与冲突处理。实验发现，现有 VLA 的零样本个性化能力有限；SFT 可以学习偏好，但不同偏好的组合与泛化仍然困难。论文已被 NeurIPS 2026 接收，并公开代码和数据。

2. ForceBench: Benchmarking Force Integration for Robotic Manipulation

中文标题：ForceBench：机器人操作中的力信息融合评测基准

机构：待核实（作者：Xuan Han、Ruihai Wu、Zhibo Pang 等）。

论文链接：https://arxiv.org/abs/2610.19604

解决的问题：接触丰富的机器人操作需要准确利用力信息，但现有策略和 Benchmark 对力反馈融合能力的评测尚不统一。

创新点：提出 ForceBench，研究机器人策略对力信息的利用能力，为比较不同力感知与力控制方法提供统一评测基础。

3. CQDC-Warp: GPU-Accelerated Differentiable Quasi-Dynamic Simulation for Contact-Rich Manipulation

中文标题：CQDC-Warp：面向复杂接触操作的 GPU 加速可微准动态仿真

机构：待核实（作者：Yanchen Ding、Jiayi Chen、Li Yi、He Wang 等）。

论文链接：https://arxiv.org/abs/2610.21349

解决的问题：接触丰富的机器人操作涉及复杂接触约束和物体运动。高精度仿真计算成本较高，也会影响基于梯度的优化与策略学习。

创新点：研究 GPU 加速的可微准动态仿真，将接触操作动力学与可微计算结合，为接触丰富任务的仿真和优化提供计算工具。




Like Author

每周前沿论文manipulation&amp;locomotion · Table of Contents
Previous
【10.12-10.16前沿论文动态】人形／四足34篇
​
具身运控lab
Like
Share
Boost
Comment