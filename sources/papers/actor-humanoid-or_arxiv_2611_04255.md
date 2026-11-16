# ActOR（arXiv:2611.04255）

> 来源归档（paper）

- **标题：** ActOR: Ambient AI-Based Agentic Orchestration for Humanoid Autonomy in Operating Rooms
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.04255>（读的是 v1，2026-11-03）
- **PDF：** <https://arxiv.org/pdf/2611.04255>
- **作者与机构（论文首页）：** Han Zhang、Sicheng Lu（同等贡献）、Aleks Santari、Mathias Unberath：约翰斯·霍普金斯大学（Johns Hopkins University, Baltimore）；Jeffrey K. Jopling、Masaru Ishii：约翰斯·霍普金斯医疗机构（Johns Hopkins Medical Institutions）。通讯作者 hzhan206@jhu.edu。致谢：NSF CAREER Award No. 2239077。
- **代码：** 论文声明「The code will be made publicly available upon publication」（Conclusion 之后的 Code availability），发表前未开源；补充视频与完整 planner prompt 随代码发布（附录 S4）。
- **入库日期：** 2026-11-09
- **一句话说明：** 把墙面环境相机与人形头部相机融合成共享的物体级数字孪生，由 LLM 在其上编排预编程的导航 / 操作技能，并用执行反馈重试与重规划，在手术室术后清理任务上做仿真消融与 Unitree G1 实机验证。

## 核对范围与通读

- **核对范围：** 读了 arXiv v1 PDF 全文：摘要、第 1 节引言、第 2 节相关工作、第 3 节方法（3.1 问题形式化、3.2 共享场景表示、3.3 闭环编排含 Algorithm 1、3.4 技能实现）、第 4 节实验（4.1 设置、4.2 仿真、4.3 实机）、第 5 节讨论与局限、第 6 节结论、参考文献 [1]–[29]，以及补充材料 S1（Table S1–S3）、S2（研究条件与 Table S4）、S3（实现细节）、S4（planner prompt 节选）。图 1–4 看了图与图注；图 3 的柱 / 点位数值对照了渲染页。未读：被引文献本身；补充视频（未提供）；完整 prompt 与代码（未发布）。PDF 文本层中的数字做了自洽核对（Table S2 各格求和与 Table S1 / S3 与正文一致）。
- **通读：** 已完成

## 核心摘录

1. **问题（摘要、第 1 节）：** 手术室机器人只靠机载相机时，要先探索才能规划（视角移动且常被遮挡）；人形执行不完美，导航可能停在目标前、抓取可能失败，基于初始场景的计划会失效。作者主张用已有的环境（ambient）感知基础设施为机器人提供房间级上下文。
2. **方法（第 3 节）：**
   - 数字孪生 S_t = {o_i}，节点 o_i = (语义标签, 3D 位置, 3D 范围, 交互状态 ∈ {free, in-hand})（式 1）。
   - 感知：VLM（GPT-6 Luna，第 4.1 节）检测 → SAM2 实例掩码 → 立体深度反投影到世界系；检测与已有节点重叠最多者合并，否则新建；连续深度观测发现体积空闲则删除节点；夹爪未完全闭合时标 in-hand。（3.2）
   - 规划：LLM（GPT-6.1 Sol）输入指令、JSON 序列化的 DT、执行历史；可调单个技能、输出多步计划、或宣布完成 / 失败；技能失败或计划结束时重新获得更新后的 DT 与历史（3.3，Algorithm 1）。重试是 prompt 规则而非固定循环（S3）。
   - 技能库（Table 1）：Observe、Explore、Navigate、Pick、Place；Navigate 用环境深度投影出 2D 占用图 + A* + 基于 RL 的全身控制器（AMO [22]）；Pick 用 AnyGrasp + 3D 包围盒启发式候选，Pinocchio 做可行性与碰撞检查，IK 转关节指令。（3.4）
3. **仿真设置（4.1、S2、S3）：** MuJoCo 中两个由真实房间重建的手术室（TwinOR，OR-A 5.46×5.63 m，OR-B 5.80×5.42 m），物体资产由 SAM3D 生成；三个任务（单物取回、双物收集、整室复位），每方法每条件 60 次试验（3 任务 × 2 房间 × 10）；d_reach = 0.85 m，T_timeout = 1800 s。基线：Robot-only（DT 只来自机器人相机）、Open-loop（同 DT / planner，但首次技能失败即结束）。
4. **环境感知消融（4.2，Fig. 3a,b，Table S2/S3）：** 目标起始在机器人视野外（partially observable）时，Explore 调用 1.05 → 0.12 次 / 试验；感知相关失败 13 → 1；成功率 80.0%（48/60）对 66.7%（40/60）。目标在视野内（ego-observable）时 80.0%（48/60）对 73.3%（44/60），Explore 调用 0.00 对 0.92（Fig. 3b）。作者自述：逐任务置信区间重叠，需要更大样本才能确认增益。
5. **闭环反馈消融（4.2，Fig. 3c，Table S1）：** 名义执行 83.3%（50/60）对 Open-loop 46.7%（28/60）；注入扰动（导航误差停在目标前 1.0–2.0 m、目标位移至少 0.5 m 且 DT 不更新、首次抓取偏移 5–8 cm 三选一）后 51.7%（31/60）对 8.3%（5/60）；按类型：导航误差 15/24 对 2/24，物体搬移 11/18 对 3/18，抓取偏差 5/18 对 0/18；多物收集 12/20 对 0/20。
6. **实机（4.3，Fig. 4）：** Unitree G1 EDU（29 DoF，Dex1-1 夹爪，头部 ZED Mini，墙面 ZED X + Jetson Orin），模拟手术室含手术台、移动 CT、手术机器人、物资台；20 次单瓶丢弃试验成功 14 次（70%），成功试验中均由环境视角定位瓶子、未调用 Explore；6 次失败中 5 次源于感知（VLM 漏检 / 误识别），1 次为重试后仍抓取失败。另有多物体探索性测试（搬动垃圾桶或新增一瓶后重规划），无成功率统计。
7. **局限（第 5 节，作者自述）：** 检测与规划依赖专有 API 模型，涉及临床隐私；VLM 在杂乱场景仍不可靠，建议改用 OR 物体的闭集检测器；腿式人形带来电机过热与操作精度下降，目前只能操作较大物体；操作仅限 pick-and-place。

## 对周报「具身运控lab」条目的核对

- 机构：周报写「约翰斯·霍普金斯大学」，与论文首页一致（另含 Johns Hopkins Medical Institutions）。
- 解决的问题：与论文一致（引言「onboard camera provides only a moving and frequently occluded view」）；论文还强调执行不完美、计划会失效。
- 创新点：「墙面相机 + 第一视角融合建立物体级数字孪生」「LLM 编排预定义技能」「依据执行反馈重规划」均与论文一致；周报未含数字。论文的结论性数字见上文核心摘录 4–6。注意论文贡献表述为「shared scene representation」「closed-loop execution mechanism」「消融 + 实机验证」，不是新的学习算法。

## 对 wiki 的映射

- [paper-actor-humanoid-or](../../wiki/entities/paper-actor-humanoid-or.md)
