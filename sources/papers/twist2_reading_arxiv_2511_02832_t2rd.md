# TWIST2 论文通读（便携全身人形数据采集系统）

> 来源归档（原文通读 · 支撑 `wiki/entities/paper-twist2.md`）

- **标题：** TWIST2: Scalable, Portable, and Holistic Humanoid Data Collection System
- **作者：** Yanjie Ze, Siheng Zhao, Weizhuo Wang, Angjoo Kanazawa, Rocky Duan, Pieter Abbeel, Guanya Shi, Jiajun Wu, C. Karen Liu（Amazon FAR / Stanford / USC / UC Berkeley / CMU）
- **类型：** paper / humanoid / teleoperation / data-collection
- **原文：** <https://arxiv.org/abs/2511.02832>（2026-11 通过 arXiv PDF v1 获取全文）
- **一句话说明：** 用 PICO 4U + 2 个 Motion Tracker 的便携无动捕方案取得全身人体姿态，配 250 美元 2-DoF 颈做主动 egocentric 立体视觉，经 GMR 重定向与 RL 通用运动跟踪器驱动 Unitree G1；采到的示范再训练 Diffusion Policy 作高层 visuomotor 策略。
- **核对范围：** 通读摘要、§I–§VI（系统设计 §III、实验 §IV、结论与局限 §V、规模化讨论 §VI）、Table I–III、Fig. 1–15 图注及参考文献列表；该版本无独立附录。视频与项目页、代码仓库未在本档核对。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **定位（§I，Table I）：** 现有人形遥操作分三类——上下半身解耦（HOMIE、MobileTV）、部分全身（AMO、CLONE，腿只跟踪底盘速度）、完整全身（HumanPlus、TWIST）。TWIST2 声称首个同时具备便携、免标定、单操作员、全身跟踪、egocentric 遥操作、足控、腕控的系统。
- **分层问题形式化（§III-A）：** 低层 π_low 为任务无关的运动跟踪器，指令 p_cmd = [ẋ,ẏ 根速度, z 根高, 横滚/俯仰, 偏航角速度, 全身关节位置]，输入含 IMU 与编码器状态，50 Hz 输出关节位置目标，由 PD 跟踪。高层 π_high 有两种：遥操作（人 + 重定向器）与 visuomotor 自主策略，二者都输出同一 p_cmd。接口用相对根平移/旋转而非绝对位姿，避免依赖全局状态估计；用全身关节位置而非仅根速度，从而支持腿部操作与舞蹈。
- **硬件（§III-B）：** G1 共 29 DoF + 两只 7 DoF Dex31 手；可拆装颈用 2 个 Dynamixel XC330-T288 做 yaw/pitch，3D 打印，成本 250 美元，ZED Mini 立体相机另 400 美元；附 MuJoCo 模型；可不拆原头与 LiDAR 安装；制作了 3 套验证易复现。
- **人体数据源（§III-C）：** PICO 4U + 2 个绑在小腿的 PICO Motion Tracker（2 追踪器模式比更多追踪器更稳），约 1000 美元，经 XRoboToolkit 以 100 Hz 流出，约 1 分钟完成设置，无需额外第三视角相机（对比 Vive Tracker）。
- **重定向（§III-D）：** 改造 GMR 的第二阶段：下半身同时优化位置与旋转，上半身仅优化旋转，位置均在骨盆系下度量，以减少脚滑并容忍全局位姿跳变；颈部由头与脊柱相对旋转 R_rel 取 yaw=atan2(r21,r11)、pitch=asin(−r31)；Dex31 当作平行夹爪，用手柄按键控制抓取标量 α，在 open/close 两组位姿间插值（含力抓与捏取两套）。
- **低层跟踪器（§III-E）：** 单阶段训练（非 teacher-student）；约 2 万条动作（GMR 重定向 7k + TWIST 原数据 13k，来源 AMASS、OMOMO、自有 MoCap），另采 73 条 PICO 动作弥合域差；奖励 r = r_track + r_reg，r_track = exp(−α‖p_cmd−p_cur‖)；PPO 训练，网络为卷积历史编码器 + MLP，把历史本体与历史参考压成紧凑隐向量。
- **采集系统（§III-F）：** 指令经 Redis 送达 π_low；立体图像经 GStreamer h265 到 PICO、经 ZMQ JPEG 到采集进程，自定义 shader 调瞳距并设焦点约 3.3 英尺；手柄摇杆实现开始/结束/暂停/手开合，使单人即可完成全流程；暂停恢复时从末位姿插值到目标避免跳变；全模块高于 50 Hz，总延迟低于 0.1 s（TWIST 为 0.5 s）；GUI 后处理切分 episode、去空闲动作、滤失败。
- **visuomotor 策略（§III-G）：** 输入 360×640 RGB 降到 224×224，本体输入用历史指令 p_cmd 而非原始状态（解耦高低层并抑制误差累积）；动作空间同 p_cmd；Diffusion Policy（1D 卷积）+ R3M 预训练 ResNet-18，预测 64 步（约 2 s），执行其中 48 步；本体加 10% 高斯噪声，视觉用随机裁剪/旋转/颜色抖动；转 ONNX 后在单张 RTX 4090 上约 20 Hz 推理。
- **长时程遥操作（§IV-A）：** 叠毛巾（连续叠 3 条，瓶颈是电机过热等鲁棒性）、搬篮穿门，强调主动 egocentric 感知与平滑全身跟踪是关键。
- **采集效率（§IV-B，Table II）：** 双手操作 18.5 min 采 98 条、成功率 100%、均 11 s/条；移动操作 19.5 min 采 46 条、100%、均 25 s/条（摘要与引言口径为约 100 条/15–20 min）。
- **用户研究（Table III，采 10 条示范）：** 完整系统新手 10/12、专家 10/11，平均 67.8 s；去立体 98.3 s；去颈 112.3 s 且新手仅 7/17 成功；去 egocentric（第三人称直通）专家 43.0 s 但需站在机器人旁，不适用于长时程移动操作。
- **自主策略（§IV-C）：** WB-Dex（弯腰从架上取杯放入地面箱，170 条示范，杯很轻，微小漂移就会抓取失败，多数情况能可靠到达杯子）；Kick-T（踢 T 形箱到目标，50 条示范，7 次中成功 6 次，只会向前踢，不能绕箱调整角度）。
- **局限与讨论（§V–§VI）：** 通用跟踪器难以应对冲刺等高动态动作；PICO 全身估计不及高端动捕，尤其无追踪器的肘与膝；倡导先标准化人形硬件（以 G1 为选择），颈插件约覆盖人类核心功能的 80%；数据全部公开于 HuggingFace 与 twist-data.github.io。

## 对 wiki 的映射

- [`wiki/entities/paper-twist2.md`](../../wiki/entities/paper-twist2.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2511.02832>
