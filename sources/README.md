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
| [x] [universal_skeleton.md](papers/universal_skeleton.md) | HOVL：异构骨架开放词汇动作识别，多粒度动作-文本对齐（arXiv:2604.17013） |
| [x] [doorman_opening_sim2real_arxiv_2512_01061.md](papers/doorman_opening_sim2real_arxiv_2512_01061.md) | DoorMan：人形纯 RGB 开门 loco-manipulation（arXiv:2512.01061，CVPR 2026）摘录与 wiki 映射 |
| [x] [crisp_real2sim_iclr2026.md](papers/crisp_real2sim_iclr2026.md) | CRISP：单目视频平面原语 Real2Sim + 接触引导人形 RL（ICLR 2026）摘录与 wiki 映射 |
| [x] [dwm_arxiv_2512_17907.md](papers/dwm_arxiv_2512_17907.md) | DWM：Dexterous World Models，场景–手条件视频扩散与混合数据（arXiv:2512.17907，CVPR 2026）摘录与 wiki 映射 |
| [x] [e_sds_arxiv_2512_16446.md](papers/e_sds_arxiv_2512_16446.md) | E-SDS：环境统计条件化 VLM 奖励 + Isaac Lab 人形感知行走 RL（arXiv:2512.16446）摘录与 wiki 映射 |
| [x] [egm_arxiv_2512_19043.md](papers/egm_arxiv_2512_19043.md) | EGM：Efficient General Mimic，Bin 采样 + CDMoE + 三阶段教师–学生人形全身 tracking（arXiv:2512.19043）摘录与 wiki 映射 |
| [x] [egoscale_arxiv_2602_16710.md](papers/egoscale_arxiv_2602_16710.md) | EgoScale：2 万小时级 egocentric 人视频预训练流式 VLA + 对齐人–机 mid-training（arXiv:2602.16710）摘录与 wiki 映射 |
| [x] [faststair_arxiv_2601_10365.md](papers/faststair_arxiv_2601_10365.md) | FastStair：DCM 并行落脚点规划引导 + 分速专家 LoRA 融合的人形高速上楼梯（arXiv:2601.10365）摘录与 wiki 映射 |
| [x] [interprior_arxiv_2602_06035.md](papers/interprior_arxiv_2602_06035.md) | InterPrior：物理 HOI 生成式控制（InterMimic+ → 变分蒸馏 → RL 微调，arXiv:2602.06035，CVPR 2026 Highlight）摘录与 wiki 映射 |
| [x] [hy_motion_arxiv_2512_23464.md](papers/hy_motion_arxiv_2512_23464.md) | HY-Motion 1.0：十亿级 DiT+流匹配文本→SMPL-H 运动（arXiv:2512.23464）摘录与 wiki 映射 |
| [x] [holomotion_arxiv_2605_15336.md](papers/holomotion_arxiv_2605_15336.md) | HoloMotion-1：混合大规模运动语料 + 稀疏 MoE Transformer + 序列级 PPO 的人形零样本全身跟踪（arXiv:2605.15336，Horizon Robotics）摘录与 wiki 映射 |
| [x] [physforge_arxiv_2605_05163.md](papers/physforge_arxiv_2605_05163.md) | PhysForge：VLM 分层物理蓝图 + KVI 协同扩散生成仿真就绪关节 3D 资产；PhysDB 约 15 万四档标注（arXiv:2605.05163，HKU MMLab / 腾讯混元等）摘录与 wiki 映射 |
| [x] [gencad_arxiv_2409_16294.md](papers/gencad_arxiv_2409_16294.md) | GenCAD：图像条件 CAD program 生成（对比学习 + 潜扩散，arXiv:2409.16294，MIT）摘录与 wiki 映射 |
| [x] [gencad3d_arxiv_2509_15246.md](papers/gencad3d_arxiv_2509_15246.md) | GenCAD-3D：点云/网格→CAD program、SynthBal 与真实扫描子集（arXiv:2509.15246，MIT/JMD）摘录与 wiki 映射 |
| [x] [lift_humanoid_arxiv_2601_21363.md](papers/lift_humanoid_arxiv_2601_21363.md) | LIFT：人形 JAX SAC 大规模预训练 + 物理知情世界模型安全微调（arXiv:2601.21363）摘录与 wiki 映射 |
| [x] [bfm_humanoid_arxiv_2509_13780.md](papers/bfm_humanoid_arxiv_2509_13780.md) | BFM：CVAE + 位级掩码 + 在线蒸馏的人形 WBC 基础模型（arXiv:2509.13780，上海 AI Lab 等）摘录与 wiki 映射 |
| [x] [barkour_arxiv_2305_14654.md](papers/barkour_arxiv_2305_14654.md) | Barkour：四足敏捷障碍课基准 + 专长 PPO + Locomotion-Transformer 蒸馏 + sim2real（arXiv:2305.14654）摘录与 wiki 映射 |
| [x] [brax_arxiv_2106_13281.md](papers/brax_arxiv_2106_13281.md) | Brax：大规模可微刚体仿真与 RL（arXiv:2106.13281，NeurIPS 2021）摘录与 wiki 映射 |
| [x] [capvector_arxiv_2605_10903.md](papers/capvector_arxiv_2605_10903.md) | CapVector：参数空间 capability vector（θ_ao−θ_ft）合并 + 下游正交正则的 VLA 微调（arXiv:2605.10903，HKUSTGZ/浙大/西湖/清华/智源等）摘录与 wiki 映射 |
| [x] [daji_arxiv_2605_14417.md](papers/daji_arxiv_2605_14417.md) | DAJI：语言条件人形控制的预期关节意图接口（DAJI-Flow + DAJI-Act，arXiv:2605.14417）摘录与 wiki 映射 |
| [x] [lingbot_map_arxiv_2604_14141.md](papers/lingbot_map_arxiv_2604_14141.md) | LingBot-Map：GCA 流式 3D 重建 + Paged KV（arXiv:2604.14141）摘录与 wiki 映射 |
| [x] [mimic_video_arxiv_2512_15692.md](papers/mimic_video_arxiv_2512_15692.md) | mimic-video：Video-Action Model（VAM），互联网视频潜计划 + 流匹配动作解码器（arXiv:2512.15692）摘录与 wiki 映射 |
| [x] [defi_arxiv_2604_16391.md](papers/defi_arxiv_2604_16391.md) | DeFI：解耦 GFDM/GIDM 前向与逆动力学预训练 + 下游扩散耦合 VLA（arXiv:2604.16391）摘录与 wiki 映射 |
| [x] [wm_robot_survey_arxiv_2605_00080.md](papers/wm_robot_survey_arxiv_2605_00080.md) | World Model for Robot Learning 综述（arXiv:2605.00080）：策略内预测 / 学习型模拟器 / 可控视频生成三线 taxonomy |
| [x] [urdd_beyond_urdf_arxiv_2512_23135.md](papers/urdd_beyond_urdf_arxiv_2512_23135.md) | URDD：Beyond URDF 通用机器人描述目录（arXiv:2512.23135）摘录与 wiki 映射 |

### repos/ — 代码仓库来源归档
| 文件 | 内容 |
|------|------|
| [mujoco.md](repos/mujoco.md) | MuJoCo 物理引擎 |
| [x] [mujoco-mjx.md](repos/mujoco-mjx.md) | MuJoCo MJX：JAX/XLA 重实现（`mujoco-mjx`） |
| [x] [brax.md](repos/brax.md) | Brax：JAX 可微物理与 RL 训练（README 维护边界与 MJX/Playground 指引） |
| [x] [boyu_ai_hands_on_rl.md](repos/boyu_ai_hands_on_rl.md) | Hands-on-RL / 蘑菇书：中文 RL 教材 Jupyter 仓（PPO/SAC/MARL 等，配套 hrl.boyuai.com） |
| [isaac_gym_isaac_lab.md](repos/isaac_gym_isaac_lab.md) | Isaac Gym / Isaac Lab |
| [pinocchio.md](repos/pinocchio.md) | Pinocchio 动力学库 |
| [crocoddyl.md](repos/crocoddyl.md) | Crocoddyl 最优控制框架 |
| [unitree.md](repos/unitree.md) | Unitree 硬件与 SDK |
| [x] [unitree_ros.md](repos/unitree_ros.md) | unitree_ros：ROS1 + Gazebo8 官方描述与关节级仿真包 |
| [x] [unitree_ros_to_real.md](repos/unitree_ros_to_real.md) | unitree_ros_to_real：ROS↔真机桥与 unitree_legged_msgs（与 unitree_ros 配套） |
| [legged_gym.md](repos/legged_gym.md) | legged_gym 训练框架 |
| [x] [leggedrobotics_robotic_world_model.md](repos/leggedrobotics_robotic_world_model.md) | robotic_world_model：ETH RSL 的 RWM / RWM-U Isaac Lab 扩展（在线 + 离线想象管线） |
| [x] [leggedrobotics_robotic_world_model_lite.md](repos/leggedrobotics_robotic_world_model_lite.md) | robotic_world_model_lite：无仿真器依赖的 RWM / RWM-U 离线训练精简仓 |
| [x] [lingbot-map.md](repos/lingbot-map.md) | LingBot-Map：Robbyant 流式 3D 重建官方仓（GCT/GCA、FlashInfer、误链勘误） |
| [x] [lucidrains_mimic_video.md](repos/lucidrains_mimic_video.md) | lucidrains/mimic-video：mimic-video / VAM 论文的非官方 PyTorch 实现索引 |
| [x] [defi-logos-robotics.md](repos/defi-logos-robotics.md) | LogosRoboticsGroup/DeFi：解耦前向/逆动力学 VLA 官方实现（arXiv:2604.16391） |
| [x] [robot_lab.md](repos/robot_lab.md) | robot_lab：基于 IsaacLab 的 RL 扩展框架，支持 26+ 机器人（四足 / 轮足 / 人形） |
| [x] [rpl_cs_ucl_sds.md](repos/rpl_cs_ucl_sds.md) | RPL-CS-UCL/SDS：See it, Do it, Sorted 四足单视频技能官方实现（与 E-SDS 同系） |
| [x] [roboto_origin.md](repos/roboto_origin.md) | Roboparty 人形机器人开源聚合入口（硬件/训练/部署/描述/固件） |
| [x] [atom01_hardware.md](repos/atom01_hardware.md) | Atom01 硬件仓库（结构/CAD/PCB/BOM） |
| [x] [atom01_deploy.md](repos/atom01_deploy.md) | Atom01 部署仓库（ROS2 驱动与上机流程） |
| [x] [atom01_train.md](repos/atom01_train.md) | Atom01 训练仓库（IsaacLab 训练与迁移） |
| [x] [atom01_description.md](repos/atom01_description.md) | Atom01 描述仓库（URDF/网格/模型） |
| [x] [atom01_firmware.md](repos/atom01_firmware.md) | Atom01 固件仓库（板端构建与通信链路） |
| [x] [axellwppr_motion_tracking.md](repos/axellwppr_motion_tracking.md) | Axellwppr/motion_tracking：GentleHumanoid 全身跟踪训练/部署（mjlab，含 VR teleop 与 ONNX sim2real） |
| [x] [amp_mjlab.md](repos/amp_mjlab.md) | AMP_mjlab：Unitree G1 统一 AMP locomotion+recovery 策略（mjlab + rsl_rl） |
| [x] [apollo-lab-yale-apollo-py.md](repos/apollo-lab-yale-apollo-py.md) | apollo-py：Apollo Toolbox Python 包骨架（与 URDD 论文配套的轻量 README 入口） |
| [x] [apollo-lab-yale-apollo-resources.md](repos/apollo-lab-yale-apollo-resources.md) | apollo-resources：URDD 机器人/环境资产与 GitHub Pages 宿主（Apollo-Lab-Yale） |
| [x] [apollo-lab-yale-apollo-rust.md](repos/apollo-lab-yale-apollo-rust.md) | apollo-rust：Rust URDF→URDD 预处理与示例输出（Apollo-Lab-Yale） |
| [x] [apollo-lab-yale-apollo-three-engine.md](repos/apollo-lab-yale-apollo-three-engine.md) | apollo-three-engine：Three.js URDD 可视化公共模块（Apollo-Lab-Yale） |
| [x] [openhelix_team_capvector.md](repos/openhelix_team_capvector.md) | OpenHelix-Team/CapVector：CapVector（arXiv:2605.10903）官方训练与评估代码入口 |
| [x] [gs_playground.md](repos/gs_playground.md) | GS-Playground：批量 3DGS 光真实感并行仿真框架，RSS 2026，10^4 FPS |
| [x] [mjlab.md](repos/mjlab.md) | mjlab：Isaac Lab API + MuJoCo Warp 轻量 GPU RL 框架（AMP_mjlab / unitree_rl_mjlab 的底层） |
| [x] [newton-physics.md](repos/newton-physics.md) | Newton Physics：Warp + MuJoCo Warp GPU 可微物理引擎（LF 托管，Disney/DeepMind/NVIDIA） |
| [x] [mjlab_playground.md](repos/mjlab_playground.md) | mjlab_playground：mjlab 任务集合（MuJoCo Playground 端口起步，含 Go1/T1 getup 等） |
| [x] [freemocap.md](repos/freemocap.md) | FreeMoCap：开源低成本多相机动捕与 GUI 平台（AGPL） |
| [x] [ubisoft-laforge-animation-dataset.md](repos/ubisoft-laforge-animation-dataset.md) | LaFAN1：Ubisoft La Forge BVH 动捕与 SIGGRAPH 2020 评估脚本（CC BY-NC-ND） |
| [x] [wbc_fsm.md](repos/wbc_fsm.md) | wbc_fsm：Unitree G1 C++ 全身控制 FSM 部署框架，ONNX + Unitree SDK2，无 ROS 依赖（ccrpRepo） |
| [x] [gr00t_visual_sim2real.md](repos/gr00t_visual_sim2real.md) | GR00T-VisualSim2Real：NVIDIA 视觉 Sim2Real 框架，VIRAL + DoorMan 双 CVPR 2026 论文，PPO Teacher + DAgger RGB Student，Unitree G1 |
| [x] [horizon_robotics_holomotion.md](repos/horizon_robotics_holomotion.md) | HoloMotion：地平线人形全身运动跟踪开源栈（GitHub + Pages 文档 + arXiv:2605.15336 + HF 权重 + Docker） |
| [x] [google_deepmind_barkour_robot.md](repos/google_deepmind_barkour_robot.md) | barkour_robot：DeepMind 敏捷四足 CAD/PCBA/装配/固件（Pigweed+EtherCAT）与 OnShape、Menagerie MJCF 官方入口索引 |
| [x] [mujoco_menagerie_google_barkour_models.md](repos/mujoco_menagerie_google_barkour_models.md) | mujoco_menagerie：`google_barkour_v0` / `google_barkour_vb` 子目录（MJCF 资产） |
| [x] [sage-sim2real-actuator-gap.md](repos/sage-sim2real-actuator-gap.md) | SAGE：Isaac Sim 重放与真机关节日志对齐，量化执行器层 sim2real gap（isaac-sim2real/sage） |
| [x] [zilize-awesome-text-to-motion.md](repos/zilize-awesome-text-to-motion.md) | awesome-text-to-motion：文本驱动单人人体运动生成综述/数据集/模型精选与 GitHub Pages 交互索引（Zilize） |
| [x] [tencent_hunyuan_hy_motion_1_0.md](repos/tencent_hunyuan_hy_motion_1_0.md) | HY-Motion-1.0：腾讯混元文本→3D 人体运动 DiT+Flow Matching 官方代码与 HF 权重入口 |
| [x] [bigai-lift-humanoid.md](repos/bigai-lift-humanoid.md) | LIFT-humanoid：BIGAI 人形 SAC 预训练 + Brax 物理知情世界模型微调开源管线 |
| [x] [nousresearch_hermes_agent.md](repos/nousresearch_hermes_agent.md) | NousResearch/hermes-agent：常驻自主代理运行时（AIAgent + 网关 + 记忆/技能闭环 + 多沙箱 + 轨迹导出，MIT） |
| [x] [obra-superpowers.md](repos/obra-superpowers.md) | obra/superpowers：编码代理可组合技能 + TDD / worktree / 子代理交付方法论（多 harness 插件） |
| [x] [caveman.md](repos/caveman.md) | JuliusBrussee/caveman：多 harness 洞穴语输出/上下文压缩技能（~65% 输出 token 宣称，MIT） |
| [x] [mattpocock-skills.md](repos/mattpocock-skills.md) | mattpocock/skills：Skills For Real Engineers（grill、CONTEXT.md、TDD、架构卫生；skills.sh 安装） |
| [x] [sensenova-skills.md](repos/sensenova-skills.md) | OpenSenseNova/SenseNova-Skills：Agent Skills 办公技能库（信息图/PPT/Excel/深度研究；Hermes/OpenClaw，MIT） |
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
| [x] [wechat_embodied_ai_lab_humanoid_rl_motion_survey.md](blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md) | 具身智能研究室：42 篇 humanoid RL 运动控制「身体系统栈」长文（Agent Reach + Camoufox；`hz9JXtJeUPRfUGzfD-pZuA`） |
| [x] [wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md) | 具身智能研究室：19 篇 AMP / 运动先验专题长文（Agent Reach + Camoufox；`YZsm3855iP3TNTTt1aou7w`） |
| [x] [fsck_superpowers_announcement_2025-10-09.md](blogs/fsck_superpowers_announcement_2025-10-09.md) | Jesse Vincent：Superpowers 发布文（skills、插件启动 hook、worktree / 子代理 / 技能压力测试叙事） |
| [x] [google-research-barkour-quadruped-agility-2023-05-26.md](blogs/google-research-barkour-quadruped-agility-2023-05-26.md) | Google Research 官方博客：Barkour 四足敏捷基准与 Locomotion-Transformer 叙事（2023-05-26） |

### sites/ — 网站与在线工具归档
| 文件 | 内容 |
|------|------|
| [x] [amass-dataset.md](sites/amass-dataset.md) | AMASS：MPI-IS 统一 SMPL 人体动捕元数据集（站点与论文索引） |
| [x] [apollo-lab-yale-apollo-resources-github-io.md](sites/apollo-lab-yale-apollo-resources-github-io.md) | apollo-lab-yale.github.io/apollo-resources：URDD 浏览器内可视化（Three.js + GitHub API 列机器人） |
| [x] [bfm4humanoid-github-io.md](sites/bfm4humanoid-github-io.md) | BFM 项目页 bfm4humanoid.github.io（Roundhouse Kick / Side Salto / VR 遥操作演示，代码 In Coming） |
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
| [x] [gentle-humanoid-axell-top.md](sites/gentle-humanoid-axell-top.md) | GentleHumanoid 项目页 gentle-humanoid.axell.top（浏览器 demo、人机/物交互与实验对比，arXiv:2511.04679） |
| [x] [gencad-github-io.md](sites/gencad-github-io.md) | GenCAD 项目页 gencad.github.io（图像条件 CAD program 生成 Demo，arXiv:2409.16294） |
| [x] [gencad3d-github-io.md](sites/gencad3d-github-io.md) | GenCAD-3D 项目页 gencad3d.github.io（点云/网格→CAD、SynthBal，arXiv:2509.15246） |
| [x] [hrl-boyuai-hands-on-rl.md](sites/hrl-boyuai-hands-on-rl.md) | 动手学强化学习在线书 hrl.boyuai.com（章节 + 在线 notebook + 课件） |
| [x] [hermes-agent-nousresearch-docs.md](sites/hermes-agent-nousresearch-docs.md) | Hermes Agent 官方站 hermes-agent.nousresearch.com（产品页 + Docusaurus 文档 + llms.txt 索引） |
| [x] [npcliu-faststair-github-io.md](sites/npcliu-faststair-github-io.md) | FastStair 项目页 npcliu.github.io/FastStair（摘要、视频区、BibTeX） |
| [x] [sirui-xu-interprior-github-io.md](sites/sirui-xu-interprior-github-io.md) | InterPrior 项目页 sirui-xu.github.io/InterPrior（能力演示、BibTeX、Inter-line 姊妹链） |
| [x] [jc-bao-spider-project-github-io.md](sites/jc-bao-spider-project-github-io.md) | SPIDER 项目页 jc-bao.github.io/spider-project（管线、交互可视化、BibTeX） |
| [x] [snuvclab-dwm-github-io.md](sites/snuvclab-dwm-github-io.md) | DWM 项目页 snuvclab.github.io/dwm（TL;DR、方法洞察、BibTeX） |
| [x] [lift-humanoid-github-io.md](sites/lift-humanoid-github-io.md) | LIFT 项目页 lift-humanoid.github.io（三阶段框架、MuJoCo Playground/Brax 视频、真机微调与零样本户外片段） |
| [x] [lingbot-map-technology-robbant.md](sites/lingbot-map-technology-robbant.md) | LingBot-Map 官方项目页 technology.robbyant.com/lingbot-map（与论文/仓库交叉索引） |
| [x] [mimic-video-github-io.md](sites/mimic-video-github-io.md) | mimic-video 项目页 mimic-video.github.io（VAM 摘要、Cosmos-Predict2 方法叙述、真机与仿真结果、BibTeX） |
| [x] [motion-tracking-axell-top.md](sites/motion-tracking-axell-top.md) | motion-tracking.axell.top：Axellwppr/motion_tracking 预训练策略浏览器演示 |
| [x] [nvidia-research-egoscale.md](sites/nvidia-research-egoscale.md) | NVIDIA Research GEAR：EgoScale 项目页 research.nvidia.com/labs/gear/egoscale（演示、管线叙述、BibTeX；GitHub 标注 Coming Soon） |
| [x] [mixamo.md](sites/mixamo.md) | Mixamo：Adobe 在线角色绑定与动画库（商业服务说明） |
| [x] [mujoco-mjx-readthedocs.md](sites/mujoco-mjx-readthedocs.md) | MuJoCo 官方文档：MJX（readthedocs） |
| [x] [nvidia-physical-ai-learning.md](sites/nvidia-physical-ai-learning.md) | NVIDIA Physical AI Learning 门户（Isaac/OpenUSD/SO-101 等自学路径索引） |
| [x] [nvidia-newton-physics.md](sites/nvidia-newton-physics.md) | NVIDIA Developer：Newton Physics 产品页（Warp、OpenUSD、Isaac Lab 集成叙事） |
| [x] [newton-physics-docs-overview.md](sites/newton-physics-docs-overview.md) | Newton 官方文档 Overview（ModelBuilder 仿真循环、多求解器、URDF/MJCF/USD） |
| [x] [tairan-he.md](sites/tairan-he.md) | Tairan He（何泰然）个人主页：CMU / NVIDIA GEAR 人形学习论文与项目总索引 |
| [x] [wm-robot-survey-ntumars.md](sites/wm-robot-survey-ntumars.md) | NTUMARS 机器人世界模型综述项目站 ntumars.github.io/wm-robot-survey（arXiv:2605.00080） |
| [x] [text-to-cad-tools.md](sites/text-to-cad-tools.md) | Zoo / KittyCAD 与文字生成 CAD、同类 API 与 AEC 工具公开链接索引 |
| [x] [wuji_robotics.md](sites/wuji_robotics.md) | 舞肌科技：F 系列 / Pan Motor 电机资料 + Wuji Hand 灵巧手（docs.wuji.tech / 招聘与媒体锚点） |

### courses/ — 课程与协议入门归档
| 文件 | 内容 |
|------|------|
| [x] [uart_rs485_serial_embedded.md](courses/uart_rs485_serial_embedded.md) | UART / RS-232 / RS-485 异步串行与机器人现场布线入门（Wikipedia、TI SLLA383 等索引） |
| [x] [motor_drive_firmware_bus_protocols.md](courses/motor_drive_firmware_bus_protocols.md) | 电机驱动器底软通信：CANopen/CiA402、CoE、私有 CAN、MIT 帧、DroneCAN 等选型索引 |
| [x] [boyuai_hands_on_rl_elites_course.md](courses/boyuai_hands_on_rl_elites_course.md) | 伯禹平台《动手学强化学习》张伟楠视频课（免费，与蘑菇书/ hrl.boyuai.com 配套） |
| [x] [nvidia_sim_to_real_so101_isaac.md](courses/nvidia_sim_to_real_so101_isaac.md) | NVIDIA：SO-101 操作臂 Sim2Real 动手课（GR00T/LeRobot/Isaac Lab、四类 gap 策略） |

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
