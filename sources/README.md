# Sources

这里是 `Robotics_Notebooks` 的原始资料层。

目标不是直接回答问题，而是作为知识库的输入来源。

## 当前资料文件

### papers/ — 论文来源归档
| 文件 | 内容 |
|------|------|
| [x] [locomotion_rl.md](papers/locomotion_rl.md) | 人形/腿足机器人 RL 论文 ingest 摘要（AMP/ASE/locomotion） |
| [x] [rl_pd_action_interface_locomotion.md](papers/rl_pd_action_interface_locomotion.md) | RL+PD 动作接口与增益设计：Digit / Cassie / 四足经典 / 可变刚度 / 扭矩控制等 10 篇索引 |
| [x] [sim2real.md](papers/sim2real.md) | Sim2Real ingest 摘要（DR/RMA/InEKF） |
| [ ] [survey_papers.md](papers/survey_papers.md) | 综述论文归档（待提炼） |
| [x] [motion_control_projects.md](papers/motion_control_projects.md) | 飞书公开文档《开源运动控制项目》及其 14 个 PDF 附件来源归档 |
| [x] [humanoid_motion_control_know_how.md](papers/humanoid_motion_control_know_how.md) | 飞书公开文档《人形机器人运动控制 Know-How》结构化来源归档 |
| [x] [imitation_learning.md](papers/imitation_learning.md) | IL ingest 摘要（DAgger/ACT/Diffusion） |
| [x] [whole_body_control.md](papers/whole_body_control.md) | WBC ingest 摘要（TSID/HQP/Crocoddyl） |
| [x] [gentlehumanoid_upper_body_compliance.md](papers/gentlehumanoid_upper_body_compliance.md) | GentleHumanoid 原始资料归档（上半身柔顺 / 接触丰富人机交互） |
| [-] [humanoid_hardware.md](papers/humanoid_hardware.md) | 人形机器人硬件论文归档（当前暂缓） |
| [x] [modern_robotics_textbook.md](papers/modern_robotics_textbook.md) | Lynch & Park《Modern Robotics》教材：李群/螺旋理论统一描述运动学/动力学/控制（13 章） |
| [x] [universal_skeleton.md](papers/universal_skeleton.md) | HOVL：异构骨架开放词汇动作识别，多粒度动作-文本对齐（arXiv:2604.17013） |
| [x] [doorman_opening_sim2real_arxiv_2512_01061.md](papers/doorman_opening_sim2real_arxiv_2512_01061.md) | DoorMan：人形纯 RGB 开门 loco-manipulation（arXiv:2512.01061，CVPR 2026）摘录与 wiki 映射 |
| [x] [faststair_arxiv_2601_10365.md](papers/faststair_arxiv_2601_10365.md) | FastStair：DCM 并行落脚点规划引导 + 分速专家 LoRA 融合的人形高速上楼梯（arXiv:2601.10365）摘录与 wiki 映射 |
| [x] [interprior_arxiv_2602_06035.md](papers/interprior_arxiv_2602_06035.md) | InterPrior：物理 HOI 生成式控制（InterMimic+ → 变分蒸馏 → RL 微调，arXiv:2602.06035，CVPR 2026 Highlight）摘录与 wiki 映射 |
| [x] [lift_humanoid_arxiv_2601_21363.md](papers/lift_humanoid_arxiv_2601_21363.md) | LIFT：人形 JAX SAC 大规模预训练 + 物理知情世界模型安全微调（arXiv:2601.21363）摘录与 wiki 映射 |

### repos/ — 代码仓库来源归档
| 文件 | 内容 |
|------|------|
| [mujoco.md](repos/mujoco.md) | MuJoCo 物理引擎 |
| [isaac_gym_isaac_lab.md](repos/isaac_gym_isaac_lab.md) | Isaac Gym / Isaac Lab |
| [pinocchio.md](repos/pinocchio.md) | Pinocchio 动力学库 |
| [crocoddyl.md](repos/crocoddyl.md) | Crocoddyl 最优控制框架 |
| [unitree.md](repos/unitree.md) | Unitree 硬件与 SDK |
| [legged_gym.md](repos/legged_gym.md) | legged_gym 训练框架 |
| [x] [leggedrobotics_robotic_world_model.md](repos/leggedrobotics_robotic_world_model.md) | robotic_world_model：ETH RSL 的 RWM / RWM-U Isaac Lab 扩展（在线 + 离线想象管线） |
| [x] [leggedrobotics_robotic_world_model_lite.md](repos/leggedrobotics_robotic_world_model_lite.md) | robotic_world_model_lite：无仿真器依赖的 RWM / RWM-U 离线训练精简仓 |
| [x] [robot_lab.md](repos/robot_lab.md) | robot_lab：基于 IsaacLab 的 RL 扩展框架，支持 26+ 机器人（四足 / 轮足 / 人形） |
| [x] [roboto_origin.md](repos/roboto_origin.md) | Roboparty 人形机器人开源聚合入口（硬件/训练/部署/描述/固件） |
| [x] [atom01_hardware.md](repos/atom01_hardware.md) | Atom01 硬件仓库（结构/CAD/PCB/BOM） |
| [x] [atom01_deploy.md](repos/atom01_deploy.md) | Atom01 部署仓库（ROS2 驱动与上机流程） |
| [x] [atom01_train.md](repos/atom01_train.md) | Atom01 训练仓库（IsaacLab 训练与迁移） |
| [x] [atom01_description.md](repos/atom01_description.md) | Atom01 描述仓库（URDF/网格/模型） |
| [x] [atom01_firmware.md](repos/atom01_firmware.md) | Atom01 固件仓库（板端构建与通信链路） |
| [x] [amp_mjlab.md](repos/amp_mjlab.md) | AMP_mjlab：Unitree G1 统一 AMP locomotion+recovery 策略（mjlab + rsl_rl） |
| [x] [gs_playground.md](repos/gs_playground.md) | GS-Playground：批量 3DGS 光真实感并行仿真框架，RSS 2026，10^4 FPS |
| [x] [mjlab.md](repos/mjlab.md) | mjlab：Isaac Lab API + MuJoCo Warp 轻量 GPU RL 框架（AMP_mjlab / unitree_rl_mjlab 的底层） |
| [x] [mjlab_playground.md](repos/mjlab_playground.md) | mjlab_playground：mjlab 任务集合（MuJoCo Playground 端口起步，含 Go1/T1 getup 等） |
| [x] [freemocap.md](repos/freemocap.md) | FreeMoCap：开源低成本多相机动捕与 GUI 平台（AGPL） |
| [x] [ubisoft-laforge-animation-dataset.md](repos/ubisoft-laforge-animation-dataset.md) | LaFAN1：Ubisoft La Forge BVH 动捕与 SIGGRAPH 2020 评估脚本（CC BY-NC-ND） |
| [x] [wbc_fsm.md](repos/wbc_fsm.md) | wbc_fsm：Unitree G1 C++ 全身控制 FSM 部署框架，ONNX + Unitree SDK2，无 ROS 依赖（ccrpRepo） |
| [x] [gr00t_visual_sim2real.md](repos/gr00t_visual_sim2real.md) | GR00T-VisualSim2Real：NVIDIA 视觉 Sim2Real 框架，VIRAL + DoorMan 双 CVPR 2026 论文，PPO Teacher + DAgger RGB Student，Unitree G1 |
| [x] [sage-sim2real-actuator-gap.md](repos/sage-sim2real-actuator-gap.md) | SAGE：Isaac Sim 重放与真机关节日志对齐，量化执行器层 sim2real gap（isaac-sim2real/sage） |
| [x] [zilize-awesome-text-to-motion.md](repos/zilize-awesome-text-to-motion.md) | awesome-text-to-motion：文本驱动单人人体运动生成综述/数据集/模型精选与 GitHub Pages 交互索引（Zilize） |
| [x] [bigai-lift-humanoid.md](repos/bigai-lift-humanoid.md) | LIFT-humanoid：BIGAI 人形 SAC 预训练 + Brax 物理知情世界模型微调开源管线 |
| [x] [obra-superpowers.md](repos/obra-superpowers.md) | obra/superpowers：编码代理可组合技能 + TDD / worktree / 子代理交付方法论（多 harness 插件） |
| [x] [cyoahs-robot-motion-editor.md](repos/cyoahs-robot-motion-editor.md) | cyoahs/robot_motion_editor：浏览器 URDF + CSV 关键帧/曲线编辑，Unitree/Seed 互转（MIT） |
| [x] [project-instinct-robot-motion-editor.md](repos/project-instinct-robot-motion-editor.md) | project-instinct/robot-motion-editor：Flask + Three.js 的 URDF + NPZ 曲线编辑与平滑（Project Instinct） |
| [x] [stanford-tml-robot-keyframe-kit.md](repos/stanford-tml-robot-keyframe-kit.md) | Stanford-TML/robot_keyframe_kit：MuJoCo + Viser 通用关键帧编辑器，LZ4/joblib 导出（MIT） |

### blogs/ — 博客来源归档
| 文件 | 内容 |
|------|------|
| [x] [claw_unitree_g1_language_annotated_motion_data.md](blogs/claw_unitree_g1_language_annotated_motion_data.md) | 微信公众号文章：CLAW 为宇树 G1 生成带语言标签的物理仿真全身运动数据 |
| [x] [ted_xiao_embodied_three_eras_primary_refs.md](blogs/ted_xiao_embodied_three_eras_primary_refs.md) | Ted Xiao 访谈编译稿涉及话题的一手文献索引（论文 / 官方博客 / 技术报告） |
| [x] [fsck_superpowers_announcement_2025-10-09.md](blogs/fsck_superpowers_announcement_2025-10-09.md) | Jesse Vincent：Superpowers 发布文（skills、插件启动 hook、worktree / 子代理 / 技能压力测试叙事） |

### sites/ — 网站与在线工具归档
| 文件 | 内容 |
|------|------|
| [x] [amass-dataset.md](sites/amass-dataset.md) | AMASS：MPI-IS 统一 SMPL 人体动捕元数据集（站点与论文索引） |
| [x] [botlab_motioncanvas.md](sites/botlab_motioncanvas.md) | 地瓜机器人 BotLab（MotionCanvas）：浏览器内 obs→ONNX→MuJoCo 节点图与 MSCP |
| [x] [doorman-humanoid-github-io.md](sites/doorman-humanoid-github-io.md) | DoorMan 项目页 doorman-humanoid.github.io（管线叙述、失败案例、BibTeX、渲染工作流链接） |
| [x] [npcliu-faststair-github-io.md](sites/npcliu-faststair-github-io.md) | FastStair 项目页 npcliu.github.io/FastStair（摘要、视频区、BibTeX） |
| [x] [sirui-xu-interprior-github-io.md](sites/sirui-xu-interprior-github-io.md) | InterPrior 项目页 sirui-xu.github.io/InterPrior（能力演示、BibTeX、Inter-line 姊妹链） |
| [x] [lift-humanoid-github-io.md](sites/lift-humanoid-github-io.md) | LIFT 项目页 lift-humanoid.github.io（三阶段框架、MuJoCo Playground/Brax 视频、真机微调与零样本户外片段） |
| [x] [mixamo.md](sites/mixamo.md) | Mixamo：Adobe 在线角色绑定与动画库（商业服务说明） |
| [x] [tairan-he.md](sites/tairan-he.md) | Tairan He（何泰然）个人主页：CMU / NVIDIA GEAR 人形学习论文与项目总索引 |
| [x] [text-to-cad-tools.md](sites/text-to-cad-tools.md) | Zoo / KittyCAD 与文字生成 CAD、同类 API 与 AEC 工具公开链接索引 |
| [x] [wuji_robotics.md](sites/wuji_robotics.md) | 舞肌科技：F 系列 / Pan Motor 电机资料 + Wuji Hand 灵巧手（docs.wuji.tech / 招聘与媒体锚点） |

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
