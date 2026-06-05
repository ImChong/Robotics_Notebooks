# Sources

这里是 `Robotics_Notebooks` 的原始资料层。

目标不是直接回答问题，而是作为知识库的输入来源。

## 当前资料文件

### papers/ — 论文来源归档
| 文件 | 内容 |
|------|------|
| [x] [locomotion_rl.md](papers/locomotion_rl.md) | 人形/腿足机器人 RL 论文 ingest 摘要（AMP/ASE/locomotion） |
| [x] [rl_pd_action_interface_locomotion.md](papers/rl_pd_action_interface_locomotion.md) | RL+PD 动作接口与增益设计：Digit / Cassie / 四足经典 / 可变刚度 / 扭矩控制等 10 篇索引 |
| [x] [sds_quadruped_arxiv_2410_11571.md](papers/sds_quadruped_arxiv_2410_11571.md) | SDS：四足单视频 VLM→奖励 + IsaacGym 闭环进化（arXiv:2410.11571），E-SDS 前序方法摘录 |
| [x] [smp.md](papers/smp.md) | SMP：可复用 score-matching 运动先验（arXiv:2512.03028，SDS/ESM/GSI、100 风格组合、G1 真机）完整摘录 |
| [x] [sim2real.md](papers/sim2real.md) | Sim2Real ingest 摘要（DR/RMA/InEKF） |
| [x] [spider_scalable_physics_informed_dexterous_retargeting.md](papers/spider_scalable_physics_informed_dexterous_retargeting.md) | SPIDER：并行物理仿真采样式重定向 + 课程式虚拟接触引导（arXiv:2511.09484）摘录与 wiki 映射 |
| [ ] [survey_papers.md](papers/survey_papers.md) | 综述论文归档（待提炼） |
| [x] [motion_control_projects.md](papers/motion_control_projects.md) | 飞书公开文档《开源运动控制项目》及其 14 个 PDF 附件来源归档 |
| [x] [humanoid_motion_control_know_how.md](papers/humanoid_motion_control_know_how.md) | 飞书公开文档《人形机器人运动控制 Know-How》结构化来源归档 |
| [x] [humannet_table1_benchmark_corpora.md](papers/humannet_table1_benchmark_corpora.md) | HumanNet 论文 Table1：代表性人视频/行为语料官方入口与规模转录 |
| [x] [imitation_learning.md](papers/imitation_learning.md) | IL ingest 摘要（DAgger/ACT/Diffusion） |
| [x] [whole_body_control.md](papers/whole_body_control.md) | WBC ingest 摘要（TSID/HQP/Crocoddyl） |
| [x] [gentlehumanoid_upper_body_compliance.md](papers/gentlehumanoid_upper_body_compliance.md) | GentleHumanoid（arXiv:2511.04679）原始资料归档；已沉淀 `wiki/methods/gentlehumanoid-motion-tracking.md` |
| [-] [humanoid_hardware.md](papers/humanoid_hardware.md) | 人形机器人硬件论文归档（当前暂缓） |
| [x] [modern_robotics_textbook.md](papers/modern_robotics_textbook.md) | Lynch & Park《Modern Robotics》教材：李群/螺旋理论统一描述运动学/动力学/控制（13 章） |
| [x] [robot_link_rotor_inertia_primary_refs.md](papers/robot_link_rotor_inertia_primary_refs.md) | 连杆 URDF 惯量 + 转子反射惯量（MuJoCo armature / Gautier–Khalil 1990）一手资料索引 |
| [x] [kalman_filter_ekf_primary_refs.md](papers/kalman_filter_ekf_primary_refs.md) | KF / EKF 一手论文与教材索引（Kalman 1960；Gelb 1974；Simon 2006 等） |
| [x] [lqr_ilqr_primary_refs.md](papers/lqr_ilqr_primary_refs.md) | LQR / iLQR 一手论文与课程索引（Bryson & Ho 1975；Li & Todorov 2004 等） |
| [x] [universal_skeleton.md](papers/universal_skeleton.md) | HOVL：异构骨架开放词汇动作识别，多粒度动作-文本对齐（arXiv:2604.17013） |
| [x] [doorman_opening_sim2real_arxiv_2512_01061.md](papers/doorman_opening_sim2real_arxiv_2512_01061.md) | DoorMan：人形纯 RGB 开门 loco-manipulation（arXiv:2512.01061，CVPR 2026）摘录与 wiki 映射 |
| [x] [crisp_real2sim_iclr2026.md](papers/crisp_real2sim_iclr2026.md) | CRISP：单目视频平面原语 Real2Sim + 接触引导人形 RL（ICLR 2026）摘录与 wiki 映射 |
| [x] [dwm_arxiv_2512_17907.md](papers/dwm_arxiv_2512_17907.md) | DWM：Dexterous World Models，场景–手条件视频扩散与混合数据（arXiv:2512.17907，CVPR 2026）摘录与 wiki 映射 |
| [x] [e_sds_arxiv_2512_16446.md](papers/e_sds_arxiv_2512_16446.md) | E-SDS：环境统计条件化 VLM 奖励 + Isaac Lab 人形感知行走 RL（arXiv:2512.16446）摘录与 wiki 映射 |
| [x] [egm_arxiv_2512_19043.md](papers/egm_arxiv_2512_19043.md) | EGM：Efficient General Mimic，Bin 采样 + CDMoE + 三阶段教师–学生人形全身 tracking（arXiv:2512.19043）摘录与 wiki 映射 |
| [x] [egoscale_arxiv_2602_16710.md](papers/egoscale_arxiv_2602_16710.md) | EgoScale：2 万小时级 egocentric 人视频预训练流式 VLA + 对齐人–机 mid-training（arXiv:2602.16710）摘录与 wiki 映射 |
| [x] [explicit_stair_geometry_arxiv_2605_09944.md](papers/explicit_stair_geometry_arxiv_2605_09944.md) | 显式楼梯几何条件化：BEV 点云 → 几何 token 条件化 PPO 的人形楼梯爬升（arXiv:2605.09944，G1 实机）摘录与 wiki 映射 |
| [x] [faststair_arxiv_2601_10365.md](papers/faststair_arxiv_2601_10365.md) | FastStair：DCM 并行落脚点规划引导 + 分速专家 LoRA 融合的人形高速上楼梯（arXiv:2601.10365）摘录与 wiki 映射 |
| [x] [interprior_arxiv_2602_06035.md](papers/interprior_arxiv_2602_06035.md) | InterPrior：物理 HOI 生成式控制（InterMimic+ → 变分蒸馏 → RL 微调，arXiv:2602.06035，CVPR 2026 Highlight）摘录与 wiki 映射 |
| [x] [hy_motion_arxiv_2512_23464.md](papers/hy_motion_arxiv_2512_23464.md) | HY-Motion 1.0：十亿级 DiT+流匹配文本→SMPL-H 运动（arXiv:2512.23464）摘录与 wiki 映射 |
| [x] [holomotion_arxiv_2605_15336.md](papers/holomotion_arxiv_2605_15336.md) | HoloMotion-1：混合大规模运动语料 + 稀疏 MoE Transformer + 序列级 PPO 的人形零样本全身跟踪（arXiv:2605.15336，Horizon Robotics）摘录与 wiki 映射 |
| [x] [homeworld_arxiv_2606_06390.md](papers/homeworld_arxiv_2606_06390.md) | HomeWorld（Kairos）：文本到 sim-ready 全屋 furnished 3D 四阶段流水线 + 300K 中国住宅平面图数据集（arXiv:2606.06390，Ace Robotics / CUHK MMLab）摘录与 wiki 映射 |
| [x] [physforge_arxiv_2605_05163.md](papers/physforge_arxiv_2605_05163.md) | PhysForge：VLM 分层物理蓝图 + KVI 协同扩散生成仿真就绪关节 3D 资产；PhysDB 约 15 万四档标注（arXiv:2605.05163，HKU MMLab / 腾讯混元等）摘录与 wiki 映射 |
| [x] [physx_omni_arxiv_2605_21572.md](papers/physx_omni_arxiv_2605_21572.md) | PhysX-Omni：统一刚体/可变形/关节体 sim-ready 3D 生成；PhysXVerse + PhysX-Bench（arXiv:2605.21572，NTU S-Lab）摘录与 wiki 映射 |
| [x] [gencad_arxiv_2409_16294.md](papers/gencad_arxiv_2409_16294.md) | GenCAD：图像条件 CAD program 生成（对比学习 + 潜扩散，arXiv:2409.16294，MIT）摘录与 wiki 映射 |
| [x] [gencad3d_arxiv_2509_15246.md](papers/gencad3d_arxiv_2509_15246.md) | GenCAD-3D：点云/网格→CAD program、SynthBal 与真实扫描子集（arXiv:2509.15246，MIT/JMD）摘录与 wiki 映射 |
| [x] [lift_humanoid_arxiv_2601_21363.md](papers/lift_humanoid_arxiv_2601_21363.md) | LIFT：人形 JAX SAC 大规模预训练 + 物理知情世界模型安全微调（arXiv:2601.21363）摘录与 wiki 映射 |
| [x] [legs_arxiv_2606_01458.md](papers/legs_arxiv_2606_01458.md) | LEGS：3DGS 无遥操作 VLA 人形 loco-manip 合成数据（arXiv:2606.01458，Stanford）摘录与 wiki 映射 |
| [x] [splitadapter_arxiv_2606_03297.md](papers/splitadapter_arxiv_2606_03297.md) | SplitAdapter：负载感知因子化适配的人形搬箱 loco-manipulation（arXiv:2606.03297，Samsung）摘录与 wiki 映射 |
| [x] [bfm_humanoid_arxiv_2509_13780.md](papers/bfm_humanoid_arxiv_2509_13780.md) | BFM：CVAE + 位级掩码 + 在线蒸馏的人形 WBC 基础模型（arXiv:2509.13780，上海 AI Lab 等）摘录与 wiki 映射 |
| [x] [bfm_survey_arxiv_2506_20487.md](papers/bfm_survey_arxiv_2506_20487.md) | BFM 综述：人形 WBC 行为基础模型 taxonomy（arXiv:2506.20487，IEEE TPAMI 2025）摘录与 wiki 映射 |
| [x] [bfm_awesome_41_catalog.md](papers/bfm_awesome_41_catalog.md) | awesome-bfm-papers：**41 篇 BFM 论文 + 10 数据集** 独立 source 总索引（`papers/bfm_awesome_*.md`，配套微信公众号 41 篇专题） |
| [x] [ego_9_papers_catalog.md](papers/ego_9_papers_catalog.md) | Ego 9 篇专题：**9 篇第一视角论文** 独立 source 总索引（`papers/ego_survey_*.md`，配套 `4JQ1xa-cJ7J1ep_e4txNnA`） |
| [x] [shenlan_world_models_15_reference_catalog.md](papers/shenlan_world_models_15_reference_catalog.md) | 深蓝世界模型 15 项目：**15 篇开源 WM** 独立 source 总索引（`papers/shenlan_wm_survey_*.md`，配套 `KZT8sI4n7GvHWyM20wN3gg`） |
| [x] [php_parkour_arxiv_2602_15827.md](papers/php_parkour_arxiv_2602_15827.md) | PHP：motion matching 长程跑酷参考 + DAgger+PPO 深度多技能策略（arXiv:2602.15827，RSS 2026，Amazon FAR）摘录与 wiki 映射 |
| [x] [omniretarget_arxiv_2509_26633.md](papers/omniretarget_arxiv_2509_26633.md) | OmniRetarget：interaction mesh 交互保留重定向与数据增广（arXiv:2509.26633；PHP 上游）摘录与 wiki 映射 |
| [x] [humanoid_rl_stack_42_catalog.md](papers/humanoid_rl_stack_42_catalog.md) | 具身智能研究室 42 篇 humanoid RL 身体系统栈：独立 `humanoid_rl_stack_*` source + `paper-hrl-stack-*` 实体总索引 |
| [x] [humanoid_amp_survey_19_catalog.md](papers/humanoid_amp_survey_19_catalog.md) | 具身智能研究室 19 篇 AMP 运动先验：独立 `humanoid_amp_survey_*` source + `paper-amp-survey-*` 实体总索引 |
| [x] [bifrost_umi_arxiv_2605_03452.md](papers/bifrost_umi_arxiv_2605_03452.md) | BifrostUMI：无机器人示范 + 扩散高层 + SKR + G1 全身 visuomotor（arXiv:2605.03452，BAAI Aether）摘录与 wiki 映射 |
| [x] [barkour_arxiv_2305_14654.md](papers/barkour_arxiv_2305_14654.md) | Barkour：四足敏捷障碍课基准 + 专长 PPO + Locomotion-Transformer 蒸馏 + sim2real（arXiv:2305.14654）摘录与 wiki 映射 |
| [x] [bam_extended_friction_servos_arxiv_2410_08650.md](papers/bam_extended_friction_servos_arxiv_2410_08650.md) | BAM：舵机扩展摩擦 M1–M6 + 摆锤辨识 + MuJoCo 2R 验证（arXiv:2410.08650，ICRA 2025）摘录与 wiki 映射 |
| [x] [brax_arxiv_2106_13281.md](papers/brax_arxiv_2106_13281.md) | Brax：大规模可微刚体仿真与 RL（arXiv:2106.13281，NeurIPS 2021）摘录与 wiki 映射 |
| [x] [capvector_arxiv_2605_10903.md](papers/capvector_arxiv_2605_10903.md) | CapVector：参数空间 capability vector（θ_ao−θ_ft）合并 + 下游正交正则的 VLA 微调（arXiv:2605.10903，HKUSTGZ/浙大/西湖/清华/智源等）摘录与 wiki 映射 |
| [x] [daji_arxiv_2605_14417.md](papers/daji_arxiv_2605_14417.md) | DAJI：语言条件人形控制的预期关节意图接口（DAJI-Flow + DAJI-Act，arXiv:2605.14417）摘录与 wiki 映射 |
| [x] [lingbot_map_arxiv_2604_14141.md](papers/lingbot_map_arxiv_2604_14141.md) | LingBot-Map：GCA 流式 3D 重建 + Paged KV（arXiv:2604.14141）摘录与 wiki 映射 |
| [x] [mimic_video_arxiv_2512_15692.md](papers/mimic_video_arxiv_2512_15692.md) | mimic-video：Video-Action Model（VAM），互联网视频潜计划 + 流匹配动作解码器（arXiv:2512.15692）摘录与 wiki 映射 |
| [x] [defi_arxiv_2604_16391.md](papers/defi_arxiv_2604_16391.md) | DeFI：解耦 GFDM/GIDM 前向与逆动力学预训练 + 下游扩散耦合 VLA（arXiv:2604.16391）摘录与 wiki 映射 |
| [x] [extreme_parkour_arxiv_2309_14341.md](papers/extreme_parkour_arxiv_2309_14341.md) | Extreme Parkour：Go1 四足单目深度端到端跑酷 + 双重 DAgger 蒸馏（arXiv:2309.14341，ICRA 2024）摘录与 wiki 映射 |
| [x] [esi_bench_arxiv_2605_18746.md](papers/esi_bench_arxiv_2605_18746.md) | ESI-Bench：具身空间智能感知–行动环基准（OmniGibson，10/29/3081，arXiv:2605.18746）摘录与 wiki 映射 |
| [x] [wm_robot_survey_arxiv_2605_00080.md](papers/wm_robot_survey_arxiv_2605_00080.md) | World Model for Robot Learning 综述（arXiv:2605.00080）：策略内预测 / 学习型模拟器 / 可控视频生成三线 taxonomy |
| [x] [wem_arxiv_2605_19957.md](papers/wem_arxiv_2605_19957.md) | WEM：World-Ego Modeling + HTEWorld 混合导航–操作长程视频世界模型（arXiv:2605.19957，ZGCA-HMI-Lab）摘录与 wiki 映射 |
| [x] [ge_sim_2_arxiv_2605_27491.md](papers/ge_sim_2_arxiv_2605_27491.md) | GE-Sim 2.0：闭环操纵视频世界模拟器（本体状态专家 + World Judge + 加速，arXiv:2605.27491，AgibotTech）摘录与 wiki 映射 |
| [x] [tau0_wm_tech_report.md](papers/tau0_wm_tech_report.md) | τ₀-WM：统一视频–动作世界模型（5B VAM、异构掩码预训练、测试时 propose–evaluate–revise，Agibot Finch 技术报告 2026-05-31）摘录与 wiki 映射 |
| [x] [worldvln_arxiv_2605_15964.md](papers/worldvln_arxiv_2605_15964.md) | WorldVLN：空中 VLN 自回归 World Action Model + Action-aware GRPO（arXiv:2605.15964，EmbodiedCity）摘录与 wiki 映射 |
| [x] [unified_walk_run_recovery_sdamp_arxiv_2605_18611.md](papers/unified_walk_run_recovery_sdamp_arxiv_2605_18611.md) | SD-AMP：投影重力门控双判别器 AMP，G1 单策略走跑起身（arXiv:2605.18611，HKU）摘录与 wiki 映射 |
| [x] [heracles_humanoid_diffusion_arxiv_2603_27756.md](papers/heracles_humanoid_diffusion_arxiv_2603_27756.md) | Heracles：状态条件扩散中间件桥接跟踪与生成恢复（arXiv:2603.27756，X-Humanoid）摘录与 wiki 映射 |
| [x] [slowrl_arxiv_2603_17092.md](papers/slowrl_arxiv_2603_17092.md) | SLowRL：LoRA + Recovery 安全真机微调四足动态策略（arXiv:2603.17092，Go2）摘录与 wiki 映射 |
| [x] [any2any_arxiv_2605_23733.md](papers/any2any_arxiv_2605_23733.md) | Any2Any：跨具身 WBT 运动学对齐 + LoRA 动力学适配（arXiv:2605.23733，LimX）摘录与 wiki 映射 |
| [x] [urdd_beyond_urdf_arxiv_2512_23135.md](papers/urdd_beyond_urdf_arxiv_2512_23135.md) | URDD：Beyond URDF 通用机器人描述目录（arXiv:2512.23135）摘录与 wiki 映射 |

### repos/ — 代码仓库来源归档
| 文件 | 内容 |
|------|------|
| [mujoco.md](repos/mujoco.md) | MuJoCo 物理引擎 |
| [x] [mujoco-mjx.md](repos/mujoco-mjx.md) | MuJoCo MJX：JAX/XLA 重实现（`mujoco-mjx`） |
| [x] [brax.md](repos/brax.md) | Brax：JAX 可微物理与 RL 训练（README 维护边界与 MJX/Playground 指引） |
| [x] [boyu_ai_hands_on_rl.md](repos/boyu_ai_hands_on_rl.md) | Hands-on-RL / 蘑菇书：中文 RL 教材 Jupyter 仓（PPO/SAC/MARL 等，配套 hrl.boyuai.com） |
| [isaac_gym_isaac_lab.md](repos/isaac_gym_isaac_lab.md) | Isaac Gym / Isaac Lab |
| [x] [nvidia_isaac_teleop.md](repos/nvidia_isaac_teleop.md) | Isaac Teleop：NVIDIA 统一仿真/真机 XR 遥操作、retargeting 与 Isaac Lab 集成 |
| [pinocchio.md](repos/pinocchio.md) | Pinocchio 动力学库 |
| [crocoddyl.md](repos/crocoddyl.md) | Crocoddyl 最优控制框架 |
| [unitree.md](repos/unitree.md) | Unitree 硬件与 SDK |
| [x] [unitree_ros.md](repos/unitree_ros.md) | unitree_ros：ROS1 + Gazebo8 官方描述与关节级仿真包 |
| [x] [unitree_ros_to_real.md](repos/unitree_ros_to_real.md) | unitree_ros_to_real：ROS↔真机桥与 unitree_legged_msgs（与 unitree_ros 配套） |
| [x] [extreme-parkour.md](repos/extreme-parkour.md) | Extreme Parkour 官方代码（ICRA 2024；Isaac Gym + legged_gym 两阶段跑酷训练） |
| [legged_gym.md](repos/legged_gym.md) | legged_gym 训练框架 |
| [x] [leggedrobotics_robotic_world_model.md](repos/leggedrobotics_robotic_world_model.md) | robotic_world_model：ETH RSL 的 RWM / RWM-U Isaac Lab 扩展（在线 + 离线想象管线） |
| [x] [leggedrobotics_robotic_world_model_lite.md](repos/leggedrobotics_robotic_world_model_lite.md) | robotic_world_model_lite：无仿真器依赖的 RWM / RWM-U 离线训练精简仓 |
| [x] [lingbot-map.md](repos/lingbot-map.md) | LingBot-Map：Robbyant 流式 3D 重建官方仓（GCT/GCA、FlashInfer、误链勘误） |
| [x] [lucidrains_mimic_video.md](repos/lucidrains_mimic_video.md) | lucidrains/mimic-video：mimic-video / VAM 论文的非官方 PyTorch 实现索引 |
| [x] [defi-logos-robotics.md](repos/defi-logos-robotics.md) | LogosRoboticsGroup/DeFi：解耦前向/逆动力学 VLA 官方实现（arXiv:2604.16391） |
| [x] [easy_quadruped.md](repos/easy_quadruped.md) | Xzgz718/easy_quadruped：StanfordQuadruped 二次开发，Pupper 步态控制 + MuJoCo 浮动机身闭环仿真 |
| [x] [pupperv3_monorepo.md](repos/pupperv3_monorepo.md) | Nate711/pupperv3-monorepo：Pupper v3 机载 ROS 2 软件（与官方文档 ~/pupperv3-monorepo 一致） |
| [x] [esi_bench.md](repos/esi_bench.md) | ESI-Bench/ESI-Bench：OmniGibson 主动探索评测与 HF 数据集（arXiv:2605.18746） |
| [x] [robot_lab.md](repos/robot_lab.md) | robot_lab：基于 IsaacLab 的 RL 扩展框架，支持 26+ 机器人（四足 / 轮足 / 人形） |
| [x] [rpl_cs_ucl_sds.md](repos/rpl_cs_ucl_sds.md) | RPL-CS-UCL/SDS：See it, Do it, Sorted 四足单视频技能官方实现（与 E-SDS 同系） |
| [x] [roboto_origin.md](repos/roboto_origin.md) | Roboparty 人形机器人开源聚合入口（硬件/训练/部署/描述/固件） |
| [x] [openloong.md](repos/openloong.md) | OpenLoong 青龙全栈开源（Framework / Dyn-Control / 数据集 / loongOpen 组织矩阵） |
| [x] [openloong_hardware.md](repos/openloong_hardware.md) | OpenLoong-Hardware / AtomGit：青龙公版机 PDF 图纸与 v2.5 硬件说明 |
| [x] [atom01_hardware.md](repos/atom01_hardware.md) | Atom01 硬件仓库（结构/CAD/PCB/BOM） |
| [x] [atom01_deploy.md](repos/atom01_deploy.md) | Atom01 部署仓库（ROS2 驱动与上机流程） |
| [x] [atom01_train.md](repos/atom01_train.md) | Atom01 训练仓库（IsaacLab 训练与迁移） |
| [x] [atom01_description.md](repos/atom01_description.md) | Atom01 描述仓库（URDF/网格/模型） |
| [x] [atom01_firmware.md](repos/atom01_firmware.md) | Atom01 固件仓库（板端构建与通信链路） |
| [x] [open_duck_mini.md](repos/open_duck_mini.md) | Open Duck Mini：BDX 迷你双足 Hub（CAD/BOM/v2 sim2real 文档） |
| [x] [open_duck_playground.md](repos/open_duck_playground.md) | Open Duck Playground：MuJoCo Playground/MJX RL 训练与 ONNX 导出 |
| [x] [open_duck_reference_motion_generator.md](repos/open_duck_reference_motion_generator.md) | Open Duck 参考运动：Placo 参数化步态 → 模仿奖励系数 |
| [x] [open_duck_mini_runtime.md](repos/open_duck_mini_runtime.md) | Open Duck Mini Runtime：Pi Zero 2W 机载 ONNX 与 Feetech 驱动 |
| [x] [axellwppr_motion_tracking.md](repos/axellwppr_motion_tracking.md) | Axellwppr/motion_tracking：GentleHumanoid 全身跟踪训练/部署（mjlab，含 VR teleop 与 ONNX sim2real） |
| [x] [amp_mjlab.md](repos/amp_mjlab.md) | AMP_mjlab：Unitree G1 统一 AMP locomotion+recovery 策略（mjlab + rsl_rl） |
| [x] [smp_suz_tsinghua.md](repos/smp_suz_tsinghua.md) | SUZ-tsinghua/smp：Unitree G1 上 SMP（mjlab）端到端复现，预置三套 prior 与乘性 task×SMP 奖励 |
| [x] [apollo-lab-yale-apollo-py.md](repos/apollo-lab-yale-apollo-py.md) | apollo-py：Apollo Toolbox Python 包骨架（与 URDD 论文配套的轻量 README 入口） |
| [x] [apollo-lab-yale-apollo-resources.md](repos/apollo-lab-yale-apollo-resources.md) | apollo-resources：URDD 机器人/环境资产与 GitHub Pages 宿主（Apollo-Lab-Yale） |
| [x] [apollo-lab-yale-apollo-rust.md](repos/apollo-lab-yale-apollo-rust.md) | apollo-rust：Rust URDF→URDD 预处理与示例输出（Apollo-Lab-Yale） |
| [x] [apollo-lab-yale-apollo-three-engine.md](repos/apollo-lab-yale-apollo-three-engine.md) | apollo-three-engine：Three.js URDD 可视化公共模块（Apollo-Lab-Yale） |
| [x] [openhelix_team_capvector.md](repos/openhelix_team_capvector.md) | OpenHelix-Team/CapVector：CapVector（arXiv:2605.10903）官方训练与评估代码入口 |
| [x] [gs_playground.md](repos/gs_playground.md) | GS-Playground：批量 3DGS 光真实感并行仿真框架，RSS 2026，10^4 FPS |
| [x] [aholo-viewer.md](repos/aholo-viewer.md) | Aholo Viewer：Web 高性能 3DGS+Mesh，Chunked Streaming LoD（manycoretech） |
| [x] [mjlab.md](repos/mjlab.md) | mjlab：Isaac Lab API + MuJoCo Warp 轻量 GPU RL 框架（AMP_mjlab / unitree_rl_mjlab 的底层） |
| [x] [newton-physics.md](repos/newton-physics.md) | Newton Physics：Warp + MuJoCo Warp GPU 可微物理引擎（LF 托管，Disney/DeepMind/NVIDIA） |
| [x] [ppf-contact-solver.md](repos/ppf-contact-solver.md) | ppf-contact-solver：ZOZO GPU shell/solid/rod FEM+接触离线仿真（TOG 论文实现） |
| [x] [mjlab_playground.md](repos/mjlab_playground.md) | mjlab_playground：mjlab 任务集合（MuJoCo Playground 端口起步，含 Go1/T1 getup 等） |
| [x] [freemocap.md](repos/freemocap.md) | FreeMoCap：开源低成本多相机动捕与 GUI 平台（AGPL） |
| [x] [ubisoft-laforge-animation-dataset.md](repos/ubisoft-laforge-animation-dataset.md) | LaFAN1：Ubisoft La Forge BVH 动捕与 SIGGRAPH 2020 评估脚本（CC BY-NC-ND） |
| [x] [wbc_fsm.md](repos/wbc_fsm.md) | wbc_fsm：Unitree G1 C++ 全身控制 FSM 部署框架，ONNX + Unitree SDK2，无 ROS 依赖（ccrpRepo） |
| [x] [wem.md](repos/wem.md) | ZGCA-HMI-Lab/WEM：World-Ego Model 与 HTEWorld 官方代码（arXiv:2605.19957） |
| [x] [ge_sim_v2.md](repos/ge_sim_v2.md) | AgibotTech/GE-Sim-V2：Genie Envisioner World Simulator 2.0（arXiv:2605.27491；代码/权重待发布） |
| [x] [sii_research_tau_0_wm.md](repos/sii_research_tau_0_wm.md) | sii-research/tau-0-wm：τ₀-WM 官方实现（Wan-2.2 VAM 部署、HF 权重；Simulator/测试时代码待发布） |
| [x] [worldvln_embodiedcity.md](repos/worldvln_embodiedcity.md) | EmbodiedCity/WorldVLN：空中 VLN 自回归 WAM 官方代码入口（arXiv:2605.15964） |
| [x] [multirotor_uav_stack_catalog.md](repos/multirotor_uav_stack_catalog.md) | 多旋翼栈 10 仓索引：PX4、XTDrone、EGO-Planner、AirSim、Flightmare、PyBullet Gym、swarm RL、Crazyflie、MAVSDK |
| [x] [navigation_slam_autonomy_stack_catalog.md](repos/navigation_slam_autonomy_stack_catalog.md) | 导航·SLAM·自动驾驶 21 仓索引：Nav2、slam_toolbox、Cartographer、FAST-LIO、VINS、Autoware、Isaac ROS、LeRobot、OpenVLA 等 |
| [x] [navigation2.md](repos/navigation2.md) | Navigation2：ROS 2 导航框架 |
| [x] [slam_toolbox.md](repos/slam_toolbox.md) | SLAM Toolbox：2D lifelong SLAM |
| [x] [cartographer.md](repos/cartographer.md) | Google Cartographer 2D/3D SLAM |
| [x] [fast_lio.md](repos/fast_lio.md) | FAST-LIO：LiDAR-惯性里程计 |
| [x] [lio_sam.md](repos/lio_sam.md) | LIO-SAM：因子图 LiDAR-惯性 SLAM |
| [x] [autoware.md](repos/autoware.md) | Autoware 开源自动驾驶全栈 |
| [x] [orb_slam3.md](repos/orb_slam3.md) | ORB-SLAM3 视觉/视觉-惯性 SLAM |
| [x] [vins_fusion.md](repos/vins_fusion.md) | VINS-Fusion 多传感器 VIO |
| [x] [openvslam.md](repos/openvslam.md) | OpenVSLAM 模块化视觉 SLAM |
| [x] [open_vins.md](repos/open_vins.md) | OpenVINS 视觉-惯性研究平台 |
| [x] [lego_loam.md](repos/lego_loam.md) | LeGO-LOAM 地面优化激光 SLAM |
| [x] [rtabmap.md](repos/rtabmap.md) | RTAB-Map RGB-D/激光建图 |
| [x] [kimera.md](repos/kimera.md) | Kimera 语义 SLAM 套件 |
| [x] [hdl_graph_slam.md](repos/hdl_graph_slam.md) | hdl_graph_slam 3D 激光图优化 |
| [x] [voxgraph.md](repos/voxgraph.md) | voxgraph TSDF 位姿图 |
| [x] [openloong_dyn_control.md](repos/openloong_dyn_control.md) | OpenLoong-Dyn-Control：人形 MPC+WBC |
| [x] [lerobot.md](repos/lerobot.md) | Hugging Face LeRobot 具身框架 |
| [x] [openvla.md](repos/openvla.md) | OpenVLA 开源视觉-语言-动作模型 |
| [x] [mushr.md](repos/mushr.md) | MuSHR 非完整约束小车导航教学平台 |
| [x] [isaac_ros_visual_slam.md](repos/isaac_ros_visual_slam.md) | Isaac ROS cuVSLAM |
| [x] [isaac_ros_nvblox.md](repos/isaac_ros_nvblox.md) | Isaac ROS nvblox TSDF/ESDF |
| [x] [px4_autopilot.md](repos/px4_autopilot.md) | PX4-Autopilot：开源多旋翼/固定翼/VTOL 飞控与 SITL |
| [x] [mavsdk.md](repos/mavsdk.md) | MAVSDK：MAVLink 兼容系统 C++/Python API |
| [x] [ego_planner_swarm.md](repos/ego_planner_swarm.md) | ego-planner-swarm：ESDF + B-spline 单/多机局部规划 |
| [x] [airsim.md](repos/airsim.md) | Microsoft AirSim：UE/Unity 视觉无人机仿真 |
| [x] [xtdrone.md](repos/xtdrone.md) | XTDrone：PX4 + ROS + Gazebo 教学仿真平台 |
| [x] [flightmare.md](repos/flightmare.md) | Flightmare：RPG 灵活四旋翼研究仿真器 |
| [x] [gamma_world.md](repos/gamma_world.md) | nv-tlabs/Gamma-World：多智能体生成式交互世界模型官方实现（arXiv:2605.28816） |
| [x] [gym_pybullet_drones.md](repos/gym_pybullet_drones.md) | gym-pybullet-drones：Gymnasium 四旋翼 RL 环境 |
| [x] [quad_swarm_rl.md](repos/quad_swarm_rl.md) | quad-swarm-rl：多四旋翼 OpenAI Gym 环境 |
| [x] [crazyswarm2.md](repos/crazyswarm2.md) | Crazyswarm2：Crazyflie 大规模群体 ROS2 框架 |
| [x] [crazyflie_firmware.md](repos/crazyflie_firmware.md) | crazyflie-firmware：Bitcraze 微四轴机载固件 |
| [x] [gr00t_visual_sim2real.md](repos/gr00t_visual_sim2real.md) | GR00T-VisualSim2Real：NVIDIA 视觉 Sim2Real 框架，VIRAL + DoorMan 双 CVPR 2026 论文，PPO Teacher + DAgger RGB Student，Unitree G1 |
| [x] [horizon_robotics_holomotion.md](repos/horizon_robotics_holomotion.md) | HoloMotion：地平线人形全身运动跟踪开源栈（GitHub + Pages 文档 + arXiv:2605.15336 + HF 权重 + Docker） |
| [x] [homeworld.md](repos/homeworld.md) | Kairos-HomeWorld/HomeWorld：全屋 sim-ready 室内场景生成（arXiv:2606.06390；代码/数据集 Coming Soon） |
| [x] [google_deepmind_barkour_robot.md](repos/google_deepmind_barkour_robot.md) | barkour_robot：DeepMind 敏捷四足 CAD/PCBA/装配/固件（Pigweed+EtherCAT）与 OnShape、Menagerie MJCF 官方入口索引 |
| [x] [mujoco_menagerie_google_barkour_models.md](repos/mujoco_menagerie_google_barkour_models.md) | mujoco_menagerie：`google_barkour_v0` / `google_barkour_vb` 子目录（MJCF 资产） |
| [x] [sage-sim2real-actuator-gap.md](repos/sage-sim2real-actuator-gap.md) | SAGE：Isaac Sim 重放与真机关节日志对齐，量化执行器层 sim2real gap（isaac-sim2real/sage） |
| [x] [rhoban_bam.md](repos/rhoban_bam.md) | Rhoban/bam：Better Actuator Models 摆锤辨识、CMA-ES 拟合 M1–M6、MuJoCo 2R 验证（ICRA 2025 配套） |
| [x] [physx-omni.md](repos/physx-omni.md) | physx-omni/PhysX-Omni：sim-ready 物理 3D 统一生成、PhysX-Bench 评测与训练/推理脚本（arXiv:2605.21572） |
| [x] [physx-omni-physxverse.md](repos/physx-omni-physxverse.md) | Hugging Face PhysXVerse：通用 physics-grounded sim-ready 3D 数据集（约 113 GB，五维物理标注） |
| [x] [awesome_bfm_papers.md](repos/awesome_bfm_papers.md) | awesome-bfm-papers：行为基础模型（BFM）论文/项目精选列表，配套 TPAMI 2025 综述（friedrichyuan / yuanmingqi 镜像） |
| [x] [zilize-awesome-text-to-motion.md](repos/zilize-awesome-text-to-motion.md) | awesome-text-to-motion：文本驱动单人人体运动生成综述/数据集/模型精选与 GitHub Pages 交互索引（Zilize） |
| [x] [tencent_hunyuan_hy_motion_1_0.md](repos/tencent_hunyuan_hy_motion_1_0.md) | HY-Motion-1.0：腾讯混元文本→3D 人体运动 DiT+Flow Matching 官方代码与 HF 权重入口 |
| [x] [bigai-lift-humanoid.md](repos/bigai-lift-humanoid.md) | LIFT-humanoid：BIGAI 人形 SAC 预训练 + Brax 物理知情世界模型微调开源管线 |
| [x] [nousresearch_hermes_agent.md](repos/nousresearch_hermes_agent.md) | NousResearch/hermes-agent：常驻自主代理运行时（AIAgent + 网关 + 记忆/技能闭环 + 多沙箱 + 轨迹导出，MIT） |
| [x] [obra-superpowers.md](repos/obra-superpowers.md) | obra/superpowers：编码代理可组合技能 + TDD / worktree / 子代理交付方法论（多 harness 插件） |
| [x] [caveman.md](repos/caveman.md) | JuliusBrussee/caveman：多 harness 洞穴语输出/上下文压缩技能（~65% 输出 token 宣称，MIT） |
| [x] [mattpocock-skills.md](repos/mattpocock-skills.md) | mattpocock/skills：Skills For Real Engineers（grill、CONTEXT.md、TDD、架构卫生；skills.sh 安装） |
| [x] [sensenova-skills.md](repos/sensenova-skills.md) | OpenSenseNova/SenseNova-Skills：Agent Skills 办公技能库（信息图/PPT/Excel/深度研究；Hermes/OpenClaw，MIT） |
| [x] [simplefoc_arduino_foc.md](repos/simplefoc_arduino_foc.md) | SimpleFOC / Arduino-FOC：跨 MCU 开源 FOC 库与 Shield/Mini 硬件生态（BLDC/步进） |
| [x] [hxxxz0_daji.md](repos/hxxxz0_daji.md) | Hxxxz0/DAJI：语言条件人形预期关节意图官方代码（arXiv:2605.14417） |
| [x] [panniantong_agent_reach.md](repos/panniantong_agent_reach.md) | Panniantong/Agent-Reach：编码代理互联网接入脚手架（CLI + doctor + 可插拔渠道与上游工具链） |
| [x] [crisp_real2sim_repo.md](repos/crisp_real2sim_repo.md) | CRISP-Real2Sim：ICLR 2026 单目视频 Real2Sim 官方代码入口索引 |
| [x] [cyoahs-robot-motion-editor.md](repos/cyoahs-robot-motion-editor.md) | cyoahs/robot_motion_editor：浏览器 URDF + CSV 关键帧/曲线编辑，Unitree/Seed 互转（MIT） |
| [x] [project-instinct-robot-motion-editor.md](repos/project-instinct-robot-motion-editor.md) | project-instinct/robot-motion-editor：Flask + Three.js 的 URDF + NPZ 曲线编辑与平滑（Project Instinct） |
| [x] [jc-bao-spider-project.md](repos/jc-bao-spider-project.md) | jc-bao/spider-project：SPIDER 论文配套 GitHub Pages 站点源码仓 |
| [x] [stanford-tml-robot-keyframe-kit.md](repos/stanford-tml-robot-keyframe-kit.md) | Stanford-TML/robot_keyframe_kit：MuJoCo + Viser 通用关键帧编辑器，LZ4/joblib 导出（MIT） |
| [x] [snuvclab_dwm.md](repos/snuvclab_dwm.md) | snuvclab/dwm：Dexterous World Models（CVPR 2026）官方代码与复现入口索引 |
| [x] [ferdous-alam-gencad.md](repos/ferdous-alam-gencad.md) | ferdous-alam/GenCAD：图像条件 CAD program 生成官方实现（arXiv:2409.16294） |
| [x] [yunomi-git-gencad-3d.md](repos/yunomi-git-gencad-3d.md) | yunomi-git/GenCAD-3D：多模态几何→CAD、SynthBal 与 HF 数据/权重（arXiv:2509.15246） |

### blogs/ — 博客来源归档
| 文件 | 内容 |
|------|------|
| [x] [egm_themoonlight_literature_review_2512_19043.md](blogs/egm_themoonlight_literature_review_2512_19043.md) | Moonlight 社区英文导读：EGM（arXiv:2512.19043）结构化摘要（非官方） |
| [x] [claw_unitree_g1_language_annotated_motion_data.md](blogs/claw_unitree_g1_language_annotated_motion_data.md) | 微信公众号文章：CLAW 为宇树 G1 生成带语言标签的物理仿真全身运动数据 |
| [x] [ted_xiao_embodied_three_eras_primary_refs.md](blogs/ted_xiao_embodied_three_eras_primary_refs.md) | Ted Xiao 访谈编译稿涉及话题的一手文献索引（论文 / 官方博客 / 技术报告） |
| [x] [wechat_zanezhang_tesla_optimus_leg_planetary_roller_screw.md](blogs/wechat_zanezhang_tesla_optimus_leg_planetary_roller_screw.md) | 微信公众号：Zane Zhang，特斯拉 Optimus 腿部行星滚柱丝杠（PRS）选型叙事与路线对比（入库归纳） |
| [x] [wechat_jixie_robot_open_source_treasury_issue01_10_robots.md](blogs/wechat_jixie_robot_open_source_treasury_issue01_10_robots.md) | 微信公众号「机械Robot」：机器人开源宝库第01期 10 个全开源网址（策展索引 + 10 实体页） |
| [x] [wechat_jixie_robot_open_source_treasury_issue02_10_robots.md](blogs/wechat_jixie_robot_open_source_treasury_issue02_10_robots.md) | 微信公众号「机械Robot」：机器人开源宝库第02期 10 个全开源网址（Reachy2、Poppy、InMoov、Doggo/Pupper 等） |
| [x] [wechat_embodied_ai_lab_robot_world_model_training_loop.md](blogs/wechat_embodied_ai_lab_robot_world_model_training_loop.md) | 微信公众号「具身智能研究室」：机器人世界模型应进入训练闭环（编译 arXiv:2605.00080 综述；Agent Reach + Camoufox 抓取） |
| [x] [wechat_embodied_ai_lab_daji_semantic_body_interface.md](blogs/wechat_embodied_ai_lab_daji_semantic_body_interface.md) | 微信公众号「具身智能研究室」：语言控制人形缺的是语义到身体接口（编译 DAJI arXiv:2605.14417） |
| [x] [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) | 具身智能研究室：42 篇 humanoid RL 运动控制「身体系统栈」长文（Agent Reach + Camoufox；`hz9JXtJeUPRfUGzfD-pZuA`；61 篇论文已各建 `paper-hrl-stack-*` / `paper-amp-survey-*` 实体） |
| [x] [wechat_shenlan_lie_group_lie_algebra_quaternion.md](blogs/wechat_shenlan_lie_group_lie_algebra_quaternion.md) | 深蓝具身智能：《具身智能基础》专栏 01 — 李群、李代数、四元数（Agent Reach + Camoufox；`JviRH2LW-fkCHA5gY7Qflw`） |
| [x] [wechat_shenlan_3d_coordinate_transforms.md](blogs/wechat_shenlan_3d_coordinate_transforms.md) | 深蓝具身智能：《具身智能基础》专栏 02 — 三维世界坐标变换（内外参、深度、手眼；`P5Jm7bMhaTHsytHStFbbLg`） |
| [x] [wechat_shenlan_riemannian_manifold_tangent_space.md](blogs/wechat_shenlan_riemannian_manifold_tangent_space.md) | 深蓝具身智能：《具身智能基础》专栏 03 — 黎曼流形与切空间（Exp/Log、工程近似；`uFTKN5FDvlHQxOSspvxVZw`） |
| [x] [wechat_shenlan_vla_github_repro_survey_2025.md](blogs/wechat_shenlan_vla_github_repro_survey_2025.md) | 深蓝具身智能：2025 GitHub 高 star VLA 开源栈复现推荐（OpenPI、VLA-Adapter、RLinf 等 11 项；`k_i-1NEBP-lEzth19HOHkQ`） |
| [x] [wechat_shenlan_vln_repro_four_paradigms_2026.md](blogs/wechat_shenlan_vln_repro_four_paradigms_2026.md) | 深蓝具身智能：VLN 四范式新手复现（VLFM、NavGPT、NoMaD、Uni-NaVid；`AzCDukzwrfIyms_65kh1mg`） |
| [x] [wechat_shenlan_world_models_15_open_source_2026.md](blogs/wechat_shenlan_world_models_15_open_source_2026.md) | 深蓝具身智能：世界模型 15 开源项目三线地图（级联/联合/沙盒；`KZT8sI4n7GvHWyM20wN3gg`） |
| [x] [wechat_embodied_ai_lab_ego_9_papers_survey.md](blogs/wechat_embodied_ai_lab_ego_9_papers_survey.md) | 具身智能研究室：9 篇 Ego 第一视角数据入口专题（四类问题各建 `wiki/overview/ego-category-*` 图谱 hub；Agent Reach + Camoufox；`4JQ1xa-cJ7J1ep_e4txNnA`） |
| [x] [wechat_embodied_ai_lab_bfm_41_papers_survey.md](blogs/wechat_embodied_ai_lab_bfm_41_papers_survey.md) | 具身智能研究室：41 篇 BFM 运控基座技术地图（五类问题各建 `wiki/overview/bfm-category-*` 图谱 hub；`Ei32la_vo0UW9Y_QCAqB2g`） |
| [x] [wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md) | 具身智能研究室：19 篇 AMP / 运动先验专题长文（Agent Reach + Camoufox；`YZsm3855iP3TNTTt1aou7w`；见 `humanoid_amp_survey_19_catalog.md`） |
| [x] [wechat_embodied_ai_lab_legs_vla_3dgs_loco_manip.md](blogs/wechat_embodied_ai_lab_legs_vla_3dgs_loco_manip.md) | 具身智能研究室：斯坦福 LEGS / 3DGS 人形 VLA loco-manip 数据工厂策展（Agent Reach + Camoufox；`B1sYOPKg6TQwnNGs-_8NDw`；arXiv:2606.01458） |
| [x] [wechat_human_five_humanoid_hardware_101.md](blogs/wechat_human_five_humanoid_hardware_101.md) | 微信公众号 human five：《Humanoid Hardware 入门 101》四万字硬件拆解（Agent Reach + Camoufox；`10hYwFzC1EuCypFVzC6QGQ`；七类 `wiki/overview/humanoid-hardware-101-*` 图谱 hub） |
| [x] [wechat_human_five_humanoid_actuator_102.md](blogs/wechat_human_five_humanoid_actuator_102.md) | 微信公众号 human five：《Humanoid 执行器 入门 102》姊妹篇（`zinp6ulTorzfqmCR_HaI5A`；八章 `wiki/overview/humanoid-actuator-102-*` + 参考文献 catalog） |
| [x] [fsck_superpowers_announcement_2025-10-09.md](blogs/fsck_superpowers_announcement_2025-10-09.md) | Jesse Vincent：Superpowers 发布文（skills、插件启动 hook、worktree / 子代理 / 技能压力测试叙事） |
| [x] [google-research-barkour-quadruped-agility-2023-05-26.md](blogs/google-research-barkour-quadruped-agility-2023-05-26.md) | Google Research 官方博客：Barkour 四足敏捷基准与 Locomotion-Transformer 叙事（2023-05-26） |
| [x] [worldlabs_spark_2_0_streaming_3dgs.md](blogs/worldlabs_spark_2_0_streaming_3dgs.md) | World Labs：Spark 2.0 流式 3DGS（LoD splat 树、.RAD、虚拟 splat 分页）技术博客归档 |

### sites/ — 网站与在线工具归档
| 文件 | 内容 |
|------|------|
| [x] [amass-dataset.md](sites/amass-dataset.md) | AMASS：MPI-IS 统一 SMPL 人体动捕元数据集（站点与论文索引） |
| [x] [apollo-lab-yale-apollo-resources-github-io.md](sites/apollo-lab-yale-apollo-resources-github-io.md) | apollo-lab-yale.github.io/apollo-resources：URDD 浏览器内可视化（Three.js + GitHub API 列机器人） |
| [x] [php-parkour-github-io.md](sites/php-parkour-github-io.md) | PHP 项目页 php-parkour.github.io（RSS 2026、浏览器 MuJoCo demo、跑酷实机视频；配套 arXiv:2602.15827） |
| [x] [bfm4humanoid-github-io.md](sites/bfm4humanoid-github-io.md) | BFM 项目页 bfm4humanoid.github.io（Roundhouse Kick / Side Salto / VR 遥操作演示，代码 In Coming） |
| [x] [bifrost-umi-project.md](sites/bifrost-umi-project.md) | BifrostUMI 项目页 baai-aether.github.io/BifrostUMI（三层级方法、采集硬件、G1 实验、BibTeX） |
| [x] [businesswire-lingbot-map-2026-04-16.md](sites/businesswire-lingbot-map-2026-04-16.md) | Business Wire：LingBot-Map 媒体发布稿（传播侧参考，性能数字需回查论文） |
| [x] [cia_can_knowledge_can_classic_and_hs.md](sites/cia_can_knowledge_can_classic_and_hs.md) | CiA CAN knowledge：经典 CAN、HS 物理层、历史与物理层选项索引 |
| [x] [cia_can_fd_basic_idea.md](sites/cia_can_fd_basic_idea.md) | CiA：CAN FD（Flexible Data Rate）基本思想 |
| [x] [cia_canopen_overview.md](sites/cia_canopen_overview.md) | CiA：CANopen CC / CANopen FD 嵌入式网络概览 |
| [x] [cia_dronecan_uavcan.md](sites/cia_dronecan_uavcan.md) | CiA + DroneCAN：UAVCAN/Cyphal 与 DroneCAN 无人机 CAN 应用层 |
| [x] [botlab_motioncanvas.md](sites/botlab_motioncanvas.md) | 地瓜机器人 BotLab（MotionCanvas）：浏览器内 obs→ONNX→MuJoCo 节点图与 MSCP |
| [x] [crisp-real2sim-project-github-io.md](sites/crisp-real2sim-project-github-io.md) | CRISP 项目页 crisp-real2sim.github.io（交互演示、与 VideoMimic 对比、Method、BibTeX） |
| [x] [capvector-github-io.md](sites/capvector-github-io.md) | CapVector 项目页 capvector.github.io（论文 / GitHub / Hugging Face 权重集合外链索引） |
| [x] [daji-hxxxz0-github-io.md](sites/daji-hxxxz0-github-io.md) | DAJI 项目页 hxxxz0.github.io/DAJI_PAGE（预期关节意图、HumanML3D/BABEL 结果，arXiv:2605.14417） |
| [x] [doorman-humanoid-github-io.md](sites/doorman-humanoid-github-io.md) | DoorMan 项目页 doorman-humanoid.github.io（管线叙述、失败案例、BibTeX、渲染工作流链接） |
| [x] [extreme-parkour-github-io.md](sites/extreme-parkour-github-io.md) | Extreme Parkour 项目页 extreme-parkour.github.io（ICRA 2024 实机视频、clearance/航向 ablation、CoRL 2023 demo） |
| [x] [esi-bench-project.md](sites/esi-bench-project.md) | ESI-Bench 项目页 esi-bench.github.io（任务 taxonomy、Key Findings、arXiv:2605.18746） |
| [x] [mobilegym-dev.md](sites/mobilegym-dev.md) | MobileGym 官网 mobilegym.dev（Live Demo、Leaderboard、Sim-to-Real，arXiv:2605.26114） |
| [x] [shape-your-body-nico-bohlinger.md](sites/shape-your-body-nico-bohlinger.md) | Shape Your Body 项目页（VGDS 交互演示、50 机训练集，arXiv:2606.00702） |
| [x] [gentle-humanoid-axell-top.md](sites/gentle-humanoid-axell-top.md) | GentleHumanoid 项目页 gentle-humanoid.axell.top（浏览器 demo、人机/物交互与实验对比，arXiv:2511.04679） |
| [x] [heracles-humanoid-control.md](sites/heracles-humanoid-control.md) | Heracles 项目页 heracles-humanoid-control.github.io（扩散中间件演示与 BibTeX，arXiv:2603.27756） |
| [x] [gencad-github-io.md](sites/gencad-github-io.md) | GenCAD 项目页 gencad.github.io（图像条件 CAD program 生成 Demo，arXiv:2409.16294） |
| [x] [gencad3d-github-io.md](sites/gencad3d-github-io.md) | GenCAD-3D 项目页 gencad3d.github.io（点云/网格→CAD、SynthBal，arXiv:2509.15246） |
| [x] [hrl-boyuai-hands-on-rl.md](sites/hrl-boyuai-hands-on-rl.md) | 动手学强化学习在线书 hrl.boyuai.com（章节 + 在线 notebook + 课件） |
| [x] [hermes-agent-nousresearch-docs.md](sites/hermes-agent-nousresearch-docs.md) | Hermes Agent 官方站 hermes-agent.nousresearch.com（产品页 + Docusaurus 文档 + llms.txt 索引） |
| [x] [npcliu-faststair-github-io.md](sites/npcliu-faststair-github-io.md) | FastStair 项目页 npcliu.github.io/FastStair（摘要、视频区、BibTeX） |
| [x] [physx-omni-github-io.md](sites/physx-omni-github-io.md) | PhysX-Omni 项目页 physx-omni.github.io（PhysXVerse / PhysX-Bench / 实验对比，arXiv:2605.21572） |
| [x] [robotics-venues-primary-refs.md](sites/robotics-venues-primary-refs.md) | ICRA、IROS、CoRL、RSS、T-RO、IJRR、Science Robotics 官方介绍与投稿入口一手索引 |
| [x] [sirui-xu-interprior-github-io.md](sites/sirui-xu-interprior-github-io.md) | InterPrior 项目页 sirui-xu.github.io/InterPrior（能力演示、BibTeX、Inter-line 姊妹链） |
| [x] [simplefoc_documentation.md](sites/simplefoc_documentation.md) | docs.simplefoc.com：Arduino SimpleFOC 官方文档（理论、运动/扭矩环、硬件与 v2.4 发布说明） |
| [x] [jc-bao-spider-project-github-io.md](sites/jc-bao-spider-project-github-io.md) | SPIDER 项目页 jc-bao.github.io/spider-project（管线、交互可视化、BibTeX） |
| [x] [kairos-homeworld-github-io.md](sites/kairos-homeworld-github-io.md) | Kairos · HomeWorld 项目页 kairos-homeworld.github.io（四阶段全屋生成、300K/5K 数据集 teaser、具身交互 demo、BibTeX） |
| [x] [snuvclab-dwm-github-io.md](sites/snuvclab-dwm-github-io.md) | DWM 项目页 snuvclab.github.io/dwm（TL;DR、方法洞察、BibTeX） |
| [x] [lift-humanoid-github-io.md](sites/lift-humanoid-github-io.md) | LIFT 项目页 lift-humanoid.github.io（三阶段框架、MuJoCo Playground/Brax 视频、真机微调与零样本户外片段） |
| [x] [legsvla-github-io.md](sites/legsvla-github-io.md) | LEGS 项目页 legsvla.github.io（3DGS loco-manip VLA 数据管线、真机 demo；arXiv:2606.01458） |
| [x] [splitadapter-github-io.md](sites/splitadapter-github-io.md) | SplitAdapter 项目页 splitadapter.github.io（负载感知因子化适配、G1 真机 demo；arXiv:2606.03297） |
| [x] [lingbot-map-technology-robbant.md](sites/lingbot-map-technology-robbant.md) | LingBot-Map 官方项目页 technology.robbyant.com/lingbot-map（与论文/仓库交叉索引） |
| [x] [mimic-video-github-io.md](sites/mimic-video-github-io.md) | mimic-video 项目页 mimic-video.github.io（VAM 摘要、Cosmos-Predict2 方法叙述、真机与仿真结果、BibTeX） |
| [x] [motion-tracking-axell-top.md](sites/motion-tracking-axell-top.md) | motion-tracking.axell.top：Axellwppr/motion_tracking 预训练策略浏览器演示 |
| [x] [nvidia-research-egoscale.md](sites/nvidia-research-egoscale.md) | NVIDIA Research GEAR：EgoScale 项目页 research.nvidia.com/labs/gear/egoscale（演示、管线叙述、BibTeX；GitHub 标注 Coming Soon） |
| [x] [mixamo.md](sites/mixamo.md) | Mixamo：Adobe 在线角色绑定与动画库（商业服务说明） |
| [x] [mujoco-mjx-readthedocs.md](sites/mujoco-mjx-readthedocs.md) | MuJoCo 官方文档：MJX（readthedocs） |
| [x] [pupper-v3-documentation-readthedocs.md](sites/pupper-v3-documentation-readthedocs.md) | Pupper v3 官方文档站（建造/安全/规格/ROS2 monorepo/RL·VLM 与 CS 123 入口） |
| [x] [nvidia-physical-ai-learning.md](sites/nvidia-physical-ai-learning.md) | NVIDIA Physical AI Learning 门户（Isaac/OpenUSD/SO-101 等自学路径索引） |
| [x] [nvidia-newton-physics.md](sites/nvidia-newton-physics.md) | NVIDIA Developer：Newton Physics 产品页（Warp、OpenUSD、Isaac Lab 集成叙事） |
| [x] [openloong_community.md](sites/openloong_community.md) | OpenLoong 社区：青龙·公版机门户（硬件 v2.5、控制框架、数据集、文档/论坛） |
| [x] [newton-physics-docs-overview.md](sites/newton-physics-docs-overview.md) | Newton 官方文档 Overview（ModelBuilder 仿真循环、多求解器、URDF/MJCF/USD） |
| [x] [tairan-he.md](sites/tairan-he.md) | Tairan He（何泰然）个人主页：CMU / NVIDIA GEAR 人形学习论文与项目总索引 |
| [x] [wm-robot-survey-ntumars.md](sites/wm-robot-survey-ntumars.md) | NTUMARS 机器人世界模型综述项目站 ntumars.github.io/wm-robot-survey（arXiv:2605.00080） |
| [x] [wem-project.md](sites/wem-project.md) | WEM 项目页 zgca-hmi-lab.github.io/WEM（World-Ego Modeling、HTEWorld 结果表与演示，arXiv:2605.19957） |
| [x] [ge-sim-v2-project.md](sites/ge-sim-v2-project.md) | GE-Sim 2.0 项目页 ge-sim-v2.github.io（多视角闭环模拟、World Judge、长视频演示，arXiv:2605.27491） |
| [x] [tau0-wm-agibot-finch.md](sites/tau0-wm-agibot-finch.md) | τ₀-WM 项目页 finch.agibot.com/research/tau0-wm（5B 统一视频–动作 WM、异构数据与测试时闭环，2026-05-31） |
| [x] [worldvln-embodiedcity.md](sites/worldvln-embodiedcity.md) | WorldVLN 项目页 embodiedcity.github.io/WorldVLN（闭环推理、两阶段训练、室内外 UAV 与真机演示，arXiv:2605.15964） |
| [x] [worldlabs-ai.md](sites/worldlabs-ai.md) | World Labs 官网：Marble / Spark / Marble Labs；Spark 2.0 见 blogs/worldlabs_spark_2_0_streaming_3dgs.md |
| [x] [text-to-cad-tools.md](sites/text-to-cad-tools.md) | Zoo / KittyCAD 与文字生成 CAD、同类 API 与 AEC 工具公开链接索引 |
| [x] [wuji_robotics.md](sites/wuji_robotics.md) | 舞肌科技：F 系列 / Pan Motor 电机资料 + Wuji Hand 灵巧手（docs.wuji.tech / 招聘与媒体锚点） |

### courses/ — 课程与协议入门归档
| 文件 | 内容 |
|------|------|
| [x] [uart_rs485_serial_embedded.md](courses/uart_rs485_serial_embedded.md) | UART / RS-232 / RS-485 异步串行与机器人现场布线入门（Wikipedia、TI SLLA383 等索引） |
| [x] [motor_drive_firmware_bus_protocols.md](courses/motor_drive_firmware_bus_protocols.md) | 电机驱动器底软通信：CANopen/CiA402、CoE、私有 CAN、MIT 帧、DroneCAN 等选型索引 |
| [x] [welch_bishop_kalman_filter.md](courses/welch_bishop_kalman_filter.md) | Welch & Bishop KF 入门教程（UNC TR / kalmanfilter.net） |
| [x] [mit_underactuated_kalman_lqr.md](courses/mit_underactuated_kalman_lqr.md) | MIT Underactuated + Optimal Control 2025（估计 / LQR / DDP 模块） |
| [x] [boyuai_hands_on_rl_elites_course.md](courses/boyuai_hands_on_rl_elites_course.md) | 伯禹平台《动手学强化学习》张伟楠视频课（免费，与蘑菇书/ hrl.boyuai.com 配套） |
| [x] [nvidia_sim_to_real_so101_isaac.md](courses/nvidia_sim_to_real_so101_isaac.md) | NVIDIA：SO-101 操作臂 Sim2Real 动手课（GR00T/LeRobot/Isaac Lab、四类 gap 策略） |
| [x] [stanford_cs123_robotics_ai.md](courses/stanford_cs123_robotics_ai.md) | Stanford CS 123 Robotics & AI（Pupper v3 配套实验课，cs123-stanford.readthedocs.io） |

### notes/ — 原始笔记归档
| 文件 | 内容 |
|------|------|
| [know-how.md](notes/know-how.md) | 人形机器人技术框架、Know-How 文档、深蓝学院课程 |
| [legged_humanoid_rl_pd_gains.md](notes/legged_humanoid_rl_pd_gains.md) | 腿足/人形 RL 关节 Kp、Kd（刚度阻尼）开源实现与文档索引 |
| [humanoid_motion_control_know_how.md](notes/humanoid_motion_control_know_how.md) | 飞书公开文档《人形机器人运动控制 Know-How》结构化来源归档 |
| [legacy-readme-resource-map.md](notes/legacy-readme-resource-map.md) | 旧 README 完整原始内容（归档备份） |

### 根目录散文件
| 文件 | 内容 |
|------|------|
| [theory.md](theory.md) | 机器人学理论、RL 基础、控制理论课程 |
| [papers.md](papers.md) | 论文来源（GitHub + arXiv） |
| [motion.md](motion.md) | 动捕数据集与运动生成 |
| [urdf.md](urdf.md) | URDF 模型资源、可视化、开源模型 |
| [retarget.md](retarget.md) | 动作重定向、MoCap、Retarget 相关 |
| [train.md](train.md) | 训练框架（IsaacGym, IsaacLab, RL/IL 框架汇总）|
| [sim2sim.md](sim2sim.md) | 仿真到仿真：Mujoco、PyBullet、Gazebo |
| [sim2real.md](sim2real.md) | 仿真到现实：部署框架、ROS2、经验分享 |

## 使用原则

1. 新资料优先先进入 `sources/`
2. 真正沉淀后的知识，再进入 `wiki/`
3. 不把 `wiki/` 写成纯链接堆
4. 不再让根目录 `README.md` 承担所有资源导航职责

## 与 wiki 的关系

- `sources/` = 输入资料层
- `wiki/` = 结构化知识层

sources 里的内容是原材料，wiki 是提炼后的知识。
