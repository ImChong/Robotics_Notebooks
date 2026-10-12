# Grounded-Dreamer 论文来源归档

## 元数据

- **题目：** Grounded-Dreamer: Robot Learning from World Model Synthetic Data with Physical Grounding
- **作者：** Feng Jiang, Shu Chen, Yuanfei Wang, Yuan Li, Zhenhao Shen, Jiaqi Liang, Yang Chen, Lei Wei, Chen Xie, Ruihai Wu（通讯）
- **单位：** Peking University（多数作者）；Lightwheel（Chen Xie）；Hong Kong University of Science and Technology (Guangzhou)（Ruihai Wu）
- **预印本：** https://arxiv.org/abs/2610.13164
- **HTML：** https://arxiv.org/html/2610.13164v1
- **PDF：** https://arxiv.org/pdf/2610.13164
- **版本：** v1，2026-10-09，cs.RO
- **项目页：** https://fffstrong.github.io/Grounded-Dreamer/（归档：[grounded-dreamer-github-io.md](../sites/grounded-dreamer-github-io.md)）
- **代码：** 项目页写 “Coming soon”，截至 2026-10-12 未发布
- **入库日期：** 2026-10-12

## 论文要解决的问题

扩展操作策略需要大量带动作标签的多样数据：真机遥操作昂贵，仿真视觉真实度有限，人类视频缺少机器人动作标签。Grounded-Dreamer 想在**不用人类示范**的前提下，用视频世界模型合成物理可行的操作数据。

## 方法与训练信息

1. **仿真重建：** 基于 LeHome 仿真器；相机外参用 FEEPE 标定，内参直接取真实相机；桌面纹理与背景用现成 LeHome 资产。
2. **物体重建：** SAM 3D 从初始帧重建刚体网格并缩放到真实尺寸；铰接物体先用 SAM 3D 拆成部件，再由 agent 依连接规则与关节规格组装。
3. **轨迹采集：** LLM agent 依参考代码模板、任务描述和 Isaac Sim 物体元数据写采集脚本，采用基于 IK 的末端控制。
4. **交互感知掩码：** 仿真帧里只保留机械臂与交互物体，用于微调 Cosmos 和逆动力学模型（IDM）。
5. **初始帧合成：** 随机化物体位姿，把渲染物体合成到真实背景上，用 Nano Banana 2 依真实物体参考照片光照/桌布纹理增强。
6. **合成：** Cosmos-Predict 2.5（2B Video2World）在增强帧上生成回放视频，SAM 做掩码，IDM 提取动作。Cosmos 微调：lr 4.315e-5，8 张 A800，1,400–2,000 迭代，全局 batch 32；Put in Drawer 用 125 帧，其余 93 帧；单卡约 74 GB 显存。IDM：SigLIP2-Large 编码器 + 流匹配动作头，5,000–10,000 步，batch 512，lr 1e-4，BF16。
7. **闭环验证：** 把合成动作在物理仿真中回放，按任务成败标为 Optimal / Sub-optimal，只保留成功轨迹。
8. **数据规模：** 每任务 100–150 条仿真轨迹；主对比每任务 50 条合成轨迹，扩展到约 150 条后趋于饱和；约 65% 的合成初始帧可用；生成阶段平均任务成功约 45%（附录 E.3，独立的回放过滤通过率未报告）。
9. **机器人：** Franka Emika + Panda 夹爪，Intel RealSense D405，RTX 4090 推理。7 个任务：Pick、Close Drawer、Press Button、Pick and Place、Put in Drawer、Sweep Trash、Slice Sausage。策略为 Diffusion Policy（DP）与 π0.5。

## 论文报告的结果

成功率（%），顺序 Pick / Button / Close Drawer / Pick&Place / Sweep Trash / Slice Sausage / Put in Drawer；DP 与 π0.5 各 20 次（扩展实验 30 次）：

- Sim-only（DP）：10/10/0/5/5/0/0；Sim-only（π0.5）：10/15/15/5/10/5/0。
- Real-only（DP）：50/60/60/50/50/70/35；GD 数据（DP）：45/50/45/35/45/55/25；Co-training（DP）：55/65/50/45/55/60/40。
- Real-only（π0.5）：65/75/65/60/60/75/45；GD 数据（π0.5）：55/75/55/50/60/60/30；Co-training（π0.5）：70/80/65/55/75/70/45。
- Low-data Real：DP 平均 8.6%，π0.5 平均 14.3%（文中数字，与逐任务表一致）。
- RIGVid：25/20/5/5/0/0/0。
- 据逐任务表算得 7 任务平均（本归档自行计算）：DP — Real-only 53.6、GD 42.9；π0.5 — Real-only 63.6、GD 55.0。GD-only π0.5 在 Button 上与 Real-only 持平（均 75%）。
- **OOD（Table 2）：** Real-only（DP）10/5/5/0/5/5/0，Verified WM data（DP）40/50/35/30/40/45/15；Real-only（π0.5）25/20/10/20/20/15/5，Verified WM data（π0.5）45/60/55/40/60/50/30。表算得平均：DP 4.3 → 36.4，π0.5 16.4 → 48.6。正文称 Real-only DP 为 7.1%，与表算平均 4.3% 不一致；本归档只用表内数字。
- **消融：** 无交互感知掩码时 DP 在各任务 0–5%，π0.5 在 0–10%。闭环验证表（Table 4）行标 100% / 80% / 50% 且正文提到“50% 次优数据”，含义有歧义；性能随验证数据占比下降而下降（50% 行：DP 5–30%，π0.5 10–35%）。
- **扩展：** Close Drawer 与 Put in Drawer 上成功率随合成轨迹增加而上升，约 150 条趋平。

## 复现与解释边界

- 作者自述：约 65% 合成初始帧可用；规模受算力而非方法限制；参考图仍需人工采集（1 张场景总览 + 1–4 张物体视角）。
- 合成数据整体仍低于真机示范数据（Real-only）；其价值主要体现在 Co-training 与 OOD 提升。
- 流程依赖多个外部模型（SAM 3D、Cosmos-Predict 2.5、Nano Banana 2、LeHome、Isaac Sim），可复现性取决于这些组件版本。
- 附录细节只抽取到部分，未逐节核对。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-grounded-dreamer.md
- 汇总页：wiki/overview/embodied-station-34-papers-spatial-grounding-technology-map.md

## 一手来源

- <https://arxiv.org/abs/2610.13164>、<https://arxiv.org/html/2610.13164v1>
- 项目页：<https://fffstrong.github.io/Grounded-Dreamer/>
