# Fly-by-Code 论文来源归档

## 元数据

- **题目：** Fly-by-Code: Embodied Coding Agents for Aerial Manipulation with Active Visual and Physical Feedback
- **方法：** Active Feedback（AF，“View and Probe”）
- **作者：** Jaewoo Lee, Jeongyeon Seo, Sihyun Cho, Gyeongrak Choe, Yutong Wang, Bavin Saravanan, Jia-Bin Huang, Furong Huang, Sebastian Scherer, Guanya Shi, H. Jin Kim, Seungjae Lee, Dongjae Lee
- **单位（HTML）：** 首尔大学（SNU）；卡内基梅隆大学（CMU）；马里兰大学帕克分校（UMD，Furong Huang 另署 All Purpose AI）；庆熙大学（Dongjae Lee；Seungjae Lee 的单位映射在抽取文本中不清）
- **预印本：** https://arxiv.org/abs/2610.12549
- **HTML：** https://arxiv.org/html/2610.12549v1
- **版本：** v1，2026-10-08，cs.RO（另列 cs.CV、cs.LG）
- **项目页：** https://fly-by-code.github.io/（见 [fly-by-code-project.md](../sites/fly-by-code-project.md)）
- **代码：** 项目页标注 “Code coming soon”；arXiv 页未列仓库
- **读取方式：** 摘要页 + HTML 全文（含附录 C–E 部分），2026-10-12

## 要解决的问题

具身编码智能体把自然语言转成机器人程序，并用执行反馈迭代修正。空中操作让这一假设失效：载荷限制了机载相机，关键结果常看不到；强动力学耦合与薄控制裕度使接触脆弱，失败原因难以从日志判断。重新阅读同样的反馈无法找回从未采集的数据，于是需要主动采集反馈。

## 方法与训练信息

- **流程：** 智能体写任务程序并执行；环境先记录任务得分；随后 AF 阶段逐条需求判断证据是否已能判定，全部确认可返回 NO_INSPECTION_NEEDED，否则写一个独立诊断程序，在任务后状态（不重置）下于仿真中运行；之后返回 FINISH 或 REGENERATE（给出完整修订的任务程序）；环境重置后进入下一次尝试，诊断观察只保留在智能体上下文中。
- **View：** 把腕部相机移到能回答具体结果问题的视角，图像与智能体写的注记返回，分割深度位置等测量走控制台输出。
- **Probe：** 在仿真里做物理交互并测量响应，例如推动不同接触点（Align 任务）、测门的铰接、超出声称载荷的提升测试（Payload）；因任务得分已记录，Probe 可使用任务执行中不宜采用的动作。两者可合并在一个诊断程序里。
- **智能体：** 主实验 Claude Opus 5（Claude Code 式持续会话，effort high，每次试验最多 1 次初始执行 + 10 次修订）；交叉检验用 Claude Code Opus 5.5 与 Codex + GPT-6 Astra（论文命名照录）。框架改自 CaP-X，RGB 观察直接给智能体。
- **仿真与 API：** Isaac Lab，扩展 AM-Bench（实验室数字孪生与仓库场景）；仿真/真机共用机器人 API：get_observation、goto_pose、open/close_gripper，Molmo 点提示与 SAM3 分割，语义地图查询（find_objects 等），抓取规划（plan_grasp 等），以及点云/变换工具。
- **真机：** 自研全向共轴四倾转旋翼空中操作平台：8 个旋翼 + 4 个倾转关节，0.62×0.69 m，飞行质量 8.30 kg；4 自由度臂 + 并行 UMI 夹爪（最大开口 80 mm）；腕部 Intel RealSense D455；OptiTrack 100 Hz 位姿 + EKF；机载 ASUS NUC 15 Pro，离线感知用 RTX A5000 工作站；SE(3) 几何 PID 控制器。

## 论文报告的结果

- **仿真 6 任务 × 10 次试验（共 60）：** 最终成功率 CaP-X 31.7%（19/60）、CaP-X+Trace 48.3%（29/60）、AF 75%（45/60）；完成判断精度 TP/(TP+FP) 分别 43.2%（19/44）、60.4%（29/48）、90.0%（45/50）；错误宣称完成次数 25 / 19 / 5。
- **分任务成功（TP/FP/ND，AF）：** Cabinet 4/1/5，1-Obj P&P 10/0/0，2-Obj P&P 9/1/0，Payload 10/0/0，Align 7/3/0，Sort 5/0/5。Payload 成功率 AF 100% 对 CaP-X 30%、Trace 60%；2-Obj P&P AF 90% 对 40% / 60%；Cabinet AF 40% 对 10% / 0%。
- **跨模型/接口：** AF 在每个模型-任务对上不低于基线；Cabinet 下 Astra API 0%→90%，Claude Code Opus 5.5 与 Codex Astra 60%→100%；论文声明接口差异不是模型的孤立效应。“无结构化诊断”条件（原始视频 + 控制台日志）Cabinet 20–50%，Payload 0%（Claude Code Opus 5）/ 20%（Codex Astra）。
- **语义地图消融（10 次/条件）：** 2-Obj P&P 完整 AF 9/10、去语义地图 4/10；Sort 5/10 对 2/10；SAM3 调用由 14 升到 54。该消融同时去掉了地图 API、提示摘要与使用指令。
- **成本（60 次试验）：** 每次估计美元 CaP-X 9.6、CaP-X+Trace 9.4、AF 17.1；每次成功 30.4 / 19.5 / 22.8（均为 token 估算）；AF 平均轮数 6.0，对基线 7.2 / 7.0。
- **真机（每任务 5 次，代码未改动直接迁移）：** 2-Objects P&P 全任务 4/5；Cabinet 全任务 3/5。失败记录：基篮位置感知误差使锤子落到框外；夹爪卡在左门把手后运动调用被拒；抓瓶下部遮挡腕部相机导致放置失败。单次运行约 8–9 分钟。

## 复现与解释边界

- 诊断增加推理与机器人交互开销；效果受机载感知误差与不可达视角限制；Probe 只在仿真中验证，不提供真机无约束探测的安全保证。
- 成功依赖运动规划与飞控；论文不声称对未见物理环境或明显不同本体的零样本泛化。
- 真机仅 5 次/任务。未做仅 View / 仅 Probe 的单独消融。
- 飞控频率在表中写 “333 Hz (33 ms)”，二者不一致，未据此引用。
- 部分附录（A.3、A.4、B、C.2–C.4、D、E 局部）在抓取中被截断。
