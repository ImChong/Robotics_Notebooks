# Bridging Task Planning and Execution for Language-Conditioned Bimanual Manipulation（arXiv:2611.12409）

> 来源归档（paper）

- **标题：** Bridging Task Planning and Execution for Language-Conditioned Bimanual Manipulation
- **类型：** paper
- **作者 / 机构：** Emanuele Cuzzocrea*、Giuseppina Iannotti*、Luca Rossini、Arturo Laurenzi、Nikos Tsagarakis（* 同等贡献）；意大利技术研究院 HHCM 实验室；热那亚大学 DIBRIS
- **arXiv：** <https://arxiv.org/abs/2611.12409>
- **PDF：** <https://arxiv.org/pdf/2611.12409v1>（v1，2026-11-09，cs.RO）
- **项目页：** <https://advrhumanoids.github.io/bridging-planning-execution/>（项目页标注 IEEE-RAS Humanoids 2026）
- **代码：** 文章未给代码；项目页只有 Paper / arXiv / Video 入口，未见代码仓库链接（2026-11-11 静态读取项目页）
- **入库日期：** 2026-11-11
- **一句话说明：** 零样本 VLM 双臂操作框架：场景图里带「每个面是否可接近 / 可达」，动作按阶段并行调度并带前后条件，分层验证器按 repeat / modify / replan 由局部到全局恢复；在 CENTAURO 真机桌面 9 个场景上评测。

## 核对范围

- **核对范围：** arXiv:2611.12409v1 PDF 全文（摘要、I 引言、II 相关工作、III 方法 A–E、IV 实验 A–B 含表 I–IV 与图 1–6、V 结论与未来工作、参考文献列表）；该版本无附录 / 补充材料。项目页正文静态读取（TL;DR、方法四步、九场景说明、结果摘要）；项目页视频未观看，代码仓库不存在（未核实到链接）。
- **通读：** 已完成

## 开源状态

- 项目页无代码、数据或权重链接；论文未声明开源。结论：**未发现开源**（项目页静态核查，2026-11-11）。
- 论文称本工作建立在作者先前工作 [19]（ICRA 2026 Workshop 论文）之上。

## 核心摘录

1. **场景表示（III-B）：** RGB 图经 VLM 得到对象实例、属性（如易碎）、成对关系（on top of / right of / in front of / grasped by）与末端状态；FoundationPose（带 CAD 的 model-based 模式）给 6D 位姿；每个物体六个面标注可接近性（与相邻物体面距离和由末端几何得到的安全阈值比较）与可达性（该方向接近位姿是否有可行逆运动学解）。
2. **动作表示（III-C，式 4）：** a_k = (action, target, mode, contact, side)，原语为 grasp / place / push / release；目标被挡时先找替代抓取方向，仍不行则生成搬移遮挡物的动作。
3. **调度与阶段条件（III-D）：** 顺序计划重组为并行阶段；每阶段有局部目标与原子谓词形式的前 / 后条件，阶段是监督单位。
4. **分层验证（III-E）：** 每条条件取 satisfied / uncertain / violated；uncertain 先补感知；确认失败后选 repeat（1 次查询）/ modify（2）/ replan（3），并有 abort；二元验证器每次失败都重规划（3 次查询）。
5. **实验（IV）：** CENTAURO 真机，双臂各 6 自由度、Dagana 末端，QP 逆运动学，基座全程静止，头载 RealSense D435；除另说明外 VLM 为 Gemini Robotics-ER 1.6；9 场景各 20 次。
6. **结果（表 III）：** 非执行感知 + 二元验证 vs 执行感知 + 分层验证：Boxes Aside 0% 到 85%，Fragile Box 0% 到 70%，Long Horizon 0% 到 50%，Bottles on Box 70% 到 95%，Push Failure 15% 到 95%；Stacked Boxes 两者均 100%。
7. **并行调度（IV-B.3，图 6）：** 执行时间 Bottles on Box 降 40%、Long Horizon 降 11%；阶段数 5 到 3、9 到 8；非 20 次均值，且不含模型查询时间。
8. **骨干（表 IV）：** 三场景平均，Gemini Robotics-ER 1.6 为 73.3% / 41.9 s，GPT-5.2 为 5.0% / 34.6 s，GPT-o3 为 80.0% / 91.4 s。

## 对 wiki 的映射

- [paper-bridging-planning-execution-bimanual](../../wiki/entities/paper-bridging-planning-execution-bimanual.md)
