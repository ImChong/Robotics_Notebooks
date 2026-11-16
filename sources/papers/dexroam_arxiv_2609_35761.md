# DexRoam: Learning Mobile Bimanual Dexterous Manipulation from Egocentric Whole-Body Human Demonstrations

- **类型：** paper
- **标题（中文）：** DexRoam：基于第一视角人体全身示范学习移动双臂灵巧操作
- **arXiv：** <https://arxiv.org/abs/2609.35761>（v1，2026-09-28）
- **全文 HTML：** <https://arxiv.org/html/2609.35761v1>
- **项目页：** <https://dexroam.github.io/>
- **代码：** <https://github.com/zhourui9813/DexRoam>（采集与对齐）、<https://github.com/zhourui9813/DexRoam-Policy-Training>（策略训练）
- **数据：** <https://huggingface.co/datasets/zhourui9813/DexRoam_Realworld_Data>（50 条真机 episode 示例，并非论文全部训练数据）
- **作者：** Rui Zhou、Yibo Yuan、Junkai Zhao、Fangyuan Zhao、Xiaoguang Zhao、Shanghang Zhang、Sirui Han
- **机构：** 香港科技大学、北京智源人工智能研究院、北京大学、北京航空航天大学、中国科学院自动化研究所
- **年份 / 会议：** 2026 / CoRL 2026（作者项目页标注 accepted）
- **核对范围：** arXiv 2609.35761v1 官方 HTML 全文，Abstract、§I–VII、Fig. 1–9、Tables I–III、附录 A（采集/同步/图像预处理）、B（Tables IV–V 与平滑性）、C（五任务失败分析）、D（Tables VI–IX，53-D/56-D 与 GR00T/π0.5 训练配置）、E（分布统计及 Table X 评分准则）；同时逐项核验官方项目页与双代码仓 README/可执行入口。未重新执行真机实验。
- **通读：** 已完成
- **一句话说明：** 用 Quest 3 + 头戴 ZED Mini 获取双目第一视角与全身姿态，经过具身、动作语义、任务进度三阶段对齐，与机器人遥操作示教联合训练统一动作空间的移动双臂 VLA。

## 原文核心问题与方法（§I–III）

移动双臂灵巧操作要求底盘、躯干、双臂、头部和手指**连续、联合**预测动作。作者反对把行走/操作切成互不相关的阶段，或把手部运动压成离散任务级原语。

1. **人体采集（§III-A，Appendix A）：** Meta Quest 3 通过 WebXR 输出 6-DoF 头姿态、33 个身体关节与每只手 25 个关键点；头戴 ZED Mini 录制同步双目 RGB。无需环境摄像头、外部标记或穿戴式追踪器，但**并非只戴一台裸头显**。ZED 图像在头显中低延迟展示；采集约 30 Hz，VR 画面流约 20 Hz。
2. **具身对齐（§III-B1）：** 在骨盆投影到地面的局部根坐标系表达全身运动，底盘映射为平面运动，躯干/双腕使用局部姿态，头部限制为 yaw/pitch，手部经 dex-retargeting 映射到双侧 12-DoF XHand。
3. **动作语义对齐（§III-B2）：** 从绝对重定向位姿改为局部相对变换。对躯干和双腕计算 `Delta_T = inverse(T_t) @ T_(t+1)`；底盘采用局部 `(delta_x, delta_y, delta_yaw)`；头部与 XHand 保留关节空间目标。
4. **时间对齐（§III-B3）：** 将每条人体轨迹按归一化任务进度 `s in [0,1]` 重采样到该任务机器人 episode 平均长度；位置线性插值、旋转 SLERP、离散量与图像最近邻。**不改变机器人控制频率与动作块长度**。
5. **策略学习（§III-B4，Appendix D）：** 对齐后人类与机器人示教共享第一视角双目图像 + 本体状态 + 语言条件，以及统一的全身动作空间。比较 Human–Robot Cotrain、Human Pretrain + Robot FT、Human Pretrain + Cotrain FT 三种策略；最后一种整体最好。

## 动作接口（Appendix D，Tables VI–VII）

- **观测：** 左右各 `640x480` RGB、任务语言、53-D 本体状态：左右臂位姿 `9+9`、躯干位姿 `9`、左右手关节 `12+12`、头部关节 `2`。正文 §III-B4 泛称 state history，但附录明确当前实现**仅一帧 proprioception，没有显式堆叠历史**。
- **动作：** 56-D = 双手关节目标 `12+12` + 双臂相对位姿 `9+9` + 躯干相对位姿 `9` + 头部关节 `2` + 底盘局部增量 `3`。
- **底盘约束：** state 没有显式 chassis pose；action 包含 chassis 平面相对移动。
- **训练配置：** π0.5：2 GPU、30k 步、batch 4、学习率 `2.5e-5`、全参数微调；GR00T N1.7-3B：4 GPU、40k 步、batch 24、学习率 `1e-4`，冻结预训练视觉/语言 backbone，训练 projector 与 diffusion action decoder。论文报告的配置不代表社区仓库脚本在任意机器上的实测性能。

## 原文结果与实验范围（§IV、Appendix B/E）

- **五个真实任务：** Pick Chips Can、Pour Water、Throw Trash、Deliver Fruit、Push Chair & Close Laptop。每任务每训练设定 **20 次真机试验**，同时记录任务成功率（SR）与任务阶段完成度（TCS）。
- **平均 SR：** GR00T N1.7 从 **29% → 56%**（+27 个百分点）；π0.5 从 **32% → 57%**（+25 个百分点）。均为论文五项任务的平均，不是“一次示范后即学会”。
- **数据效率：** 50 条人体示教/任务固定；在文中三个数据效率任务上，GR00T N1.7 加入人体数据后，**25 条机器人示教**可达到/超过 50 条机器人示教基线；π0.5 与基线相差不超过 5 个百分点。**0 条机器人示教时成功率为 0**。
- **分布与可执行性：** 五任务 MMD 平均下降 **41.7%**、SWD 下降 **31.1%**；15/15 条人体重定向轨迹真机重放成功且无 IK 失败。重放平均躯干位置误差 12.54 mm，左臂 4.01 mm，右臂 6.30 mm（Tables I–II）。
- **消融：** 用绝对动作替代相对动作时，两任务三种训练方法的成功率均为 0%；移除任务进度时间重采样，所报告消融均值 **68.3% → 44.2%**（-24.1 个百分点），**不能与五任务 56% 混为同一统计范围**。
- **部分分布外泛化：** Pick Chips Can 新背景中仅机器人模型 **50% → 20%**，Human Pretrain + Cotrain FT **90% → 55%**；论文也评估了未见指令措辞。

## 局限与复现边界（§VII，Appendix C）

- XHand 为**位置控制，无力/触觉反馈**；轻触后闭指力不足、杯体滑落、双手提篮不稳等为作者报告的真实失败模式；底盘释放时序与前序移动误差也会导致失败。
- 采集端 Meta Quest 3 + ZED Mini 不等于零附加硬件；**机器人示教采集**另用 Quest 3S、控制器与 MANUS Pro 手套。
- 实验平台是 Astribot + 双 XHand；移植其他机器人需要重新验证重定向、可达性、坐标系和数据分布。
- 官方已公开采集/对齐和策略训练代码，但公开 HF 50 条 episode 只是 **Push Chair & Close Laptop 机器人数据示例**；不能据此声称论文全数据、全任务权重或硬件复现已齐备。

## 对 wiki 的映射

- [DexRoam 唯一论文实体（精读）](../../wiki/entities/paper-dexroam-mobile-bimanual-manipulation.md)
- [双臂操作](../../wiki/tasks/bimanual-manipulation.md)、[移动操作](../../wiki/tasks/loco-manipulation.md)
- [官方项目页归档](../sites/dexroam-github-io.md)
- [采集/对齐源码](../repos/zhourui9813-dexroam.md)、[训练源码](../repos/zhourui9813-dexroam-policy-training.md)
