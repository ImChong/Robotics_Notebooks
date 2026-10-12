# ATHLETE 论文来源归档

## 元数据

- **题目：** ATHLETE: Learning Reactive Humanoid Tennis via Trajectory-Guided Motion Matching
- **作者：** Zhihao Tian, Zifan Wang, Enzheng Zhu, Xun Yang, Jialong Qin, Mengbo Wang, Junwei Liang（通讯作者 Junwei Liang）
- **单位（HTML）：** The Hong Kong University of Science and Technology (Guangzhou)
- **预印本：** https://arxiv.org/abs/2610.13572 （v1，2026-10-09，cs.RO）
- **HTML：** https://arxiv.org/html/2610.13572v1
- **项目页：** https://hazintihoa.github.io/athlete.github.io/ （署名为 Anonymous Authors）
- **代码：** https://anonymous.4open.science/r/athlete-humanoid-tennis （项目页 Code 链接；匿名镜像；论文摘要里写的是 anonymous.4open.science/w/ATHLETE2027/，该 /w/ 路径本次访问返回 HTTP 400，/r/ 路径可访问）
- **核对日期：** 2026-10-12

## 论文要解决的问题

人形网球需要随球速变化快速适应的全身协调，而覆盖足够多接近路径、击球点和时机的动捕示范成本很高。ATHLETE 用通用运动数据加普通网球视频替代任务专用动捕。

## 方法与训练信息

- **数据：** 16 条 LAFAN1 行走/跑步序列；从一段 10 分钟网球练习视频切出 164 段击球（97 正手 + 67 反手），用 GVHMR 重建人体、GMR 重定向到机器人；Stage II 训练库为 200 条生成的参考动作（仓库另述 1,000 条动态修正参考，暂未发布）。
- **TGMM（Trajectory-Guided Motion Matching）：** 按击球类型筛选并以接触代价（高度误差权重大于平面误差）排序击球片段；用 CasADi 梯形配点 + IPOPT 解平面接近轨迹（距离 ≤1.5 m 先平移后对齐，更远则转身奔跑再对齐）；在 LAFAN1 帧上做运动匹配（足、根特征与 0.4/0.8/1.2 s 后的平面位置与朝向）；用五次幂偏移衰减做过渡；再让 BeyondMimic 风格的跟踪策略在仿真里执行合成参考，其 rollout 作为教师参考（动态修正）。
- **三阶段学习：** Stage I 时间自适应教师，PPO 训练，输出关节动作与相位加速度，输入含状态、参考姿态、目标、剩余时间，奖励为 TaskNPoint 式跟踪/击球项 + 相位时序项；Stage II 学生蒸馏，把球轨迹对齐到参考接触点，在线 DAgger 式 rollout，关闭 PPO 更新、教师控制占比逐渐降低，损失为动作蒸馏 + 参考重建 + 潜在对齐（权重 0.05，潜变量 128 维，学生融合五个历史时刻）；Stage III PPO 微调，损失 L_PPO+0.05·L_act，奖励含挥拍速度、出球速度、过网高度、落点精度等。控制 50 Hz，骨盆/躯干倾角超过 50° 终止。
- **部署：** 只保留学生编码器和动作网络，推理只用机器人与球观测、无需参考动作；域随机化沿用 LATENT（球质量/恢复系数/切向速度保持/阻力，球拍质量/质心/恢复系数）。训练为单张 RTX 4090；实时推理为笔记本 RTX 5070 Ti Laptop GPU，有线以太网，50 Hz。球拍经 3D 打印连接件装在右手；真机球为手抛（论文未描述发球机）。机器人原文仅称 humanoid，策略输出 29 维关节动作并走 Unitree DDS（仓库明确为 Unitree G1）。

## 论文报告的结果

- **仿真（Table II，300 个成对球：159 正手侧、141 反手侧，3 种子）：** 仅 Stage II 的正/反手击中率 1.26% / 5.67%，完整方法 94.97% / 95.04%；回球率 0%/0% → 91.19% / 92.20%；落点误差 5.465/5.336 m → 1.150/1.314 m；甜区距离 14.09/13.92 cm → 5.61/6.54 cm。
- **教师时序：** 500 次试验 493 次触球（98.6%），362 次（72.4%）落在截止时刻 ±50 ms 内，平均绝对误差 71.23 ms。
- **参考覆盖：** K=1、164 条动作时 TGMM 参考匹配成功率 42.64%，原始库 7.57%；200 条随机子集在 K=8 达 99.21%。
- **真机（Table III，各 30 球）：** 正手击中 90.00%（27/30）、反手 86.67%（26/30）；过网 46.67%（14/30）/ 53.33%（16/30）。对照 LATENT 回球率 90.90% / 77.78%（在场内回球判据、全尺寸场地），AdaPT 击中率 52.00% / 50.67%（数值由分项乘积估计）；论文说明协议和指标不同，ATHLETE 捕获区 10.5×5.5 m，AdaPT 约 20×8 m。
- **对打：** 与新手最长 55 回合，与职业选手最长 88 回合。

## 复现与解释边界

- 真机球位置与机器人根部位姿来自外部动捕（OptiTrack/Motive），不是机载感知。
- 捕获区有限，出球速度奖励饱和在 3.5 m/s。
- 与 LATENT、AdaPT 的数字不是同一协议，不能当作严格对比。
- 动作风格多样性的保持留作未来工作。
- 附录是否完整读到：HTML 抽取未报告附录细节，附录内容未核实。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-athlete-reactive-humanoid-tennis-tgmm.md
- 项目页归档：sources/sites/athlete-tennis.md；代码归档：sources/repos/athlete-humanoid-tennis.md
