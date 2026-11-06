# Robocasa-Mem（arXiv:2611.06475）

> 来源归档（paper）

- **标题：** Robocasa-Mem: Benchmarking Environment and Process Memory in Long-Horizon Mobile Manipulation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.06475>
- **PDF：** <https://arxiv.org/pdf/2611.06475>
- **项目页：** <https://sihhangchen.github.io/RoboCasa-Mem/>
- **代码：** <https://github.com/SiHhangChen/RoboCasa-Mem>
- **入库日期：** 2026-11-06
- **线索来源：** [只在虚拟房间练习，人形机器人走进陌生房间能做什么？](../blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md)（具身智能小站，2026-11-06）
- **一句话说明：** 后半程的正确目标由机器人先前的选择决定。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-06）：资源入口 SiHhangChen/RoboCasa-Mem 经 git ls-remote 可达；项目页 Models 按钮标 coming soon（模型权重未发布）。

## 核对范围

- **核对范围：** 通读 arXiv:2611.06475v1（2026-11-04）PDF 全文：摘要、§1–§6（基准设计、评测协议、HMAP 方法、仿真/真机实验、消融、局限）与附录 A（任务定义）、B（数据与评测细节）、C（方法与训练细节）、D（分阶段完整结果）；未逐文件阅读代码仓库，也未下载数据集。
- **通读：** 已完成

## 核心摘录（据论文原文，数字取自 Table 3–5 与附录）

- 基准：9 个 RoboCasa 移动操作任务，5 类记忆（环境记忆：空间 / 属性 / 关系；过程记忆：单任务进度 / 多任务调度）；每任务 200 条成功示范，共 1,800 条；20 Hz，3 路 224×224 RGB（左右外置 + 腕部）、37 维状态、13 维动作；专家为 MPLib 臂规划 + 闭环底盘导航的规则脚本。
- 设计判据：当前观测相同、但历史不同的两条轨迹其可接受响应集合不相交（式 1，观测混叠），以此界定「需要记忆」而非任务长短。
- Behavior-Conditioned Evaluation：多合法路径任务不预设目标，由策略先前动作推出后续约束（如 SweetCup 任一杯可加糖，之后检查是否选中同一杯）。每任务 50 个留出场景种子；指标 SR 与 SCR（各阶段成功率均值）。
- HMAP：高层 Qwen3-VL-4B 规划器递归维护实体记忆 e 与过程记忆 p，输入当前帧与 50 步前帧，输出语言子目标；低层在 π0.5 上以 8 个可学习查询对冻结 VGGT-Ω 多视角几何 token 做交叉注意力，再以 scale/shift 调制动作专家。两者分开训练，控制器用 flow matching。
- 主结果（Table 3，SR/SCR 均值）：HMAP 0.24/0.47，π0.5 0.16/0.41，MemER-style 0.02/0.25，DP 0.02/0.14，ACT 0.00/0.16。
- TempReport：SR 0.22→0.86，SCR 0.57→0.93；条件正确率 0.41→0.90。SweetCup 选对加糖杯的条件率 0.22→0.45。PutBackCup 0.68→0.70，MilkMatch 0.12→0.16；CanRecall 全部 SR 为 0。
- 过程记忆：SugarCount 两者同为 0.28，DrawerSearch HMAP 0.08 低于 π0.5 的 0.12；HeatAndInspect、FillAndRetrieve 全部方法 SR 为 0（HeatAndInspect SCR 0.32→0.50）。
- 真机（AgileX Cobot Magic，物体搬运，100 条示范，每方法 10 次）：选对目标容器 HMAP 8/10，π0.5 2/10。
- 消融（仿真 SingleMilkMatch，终端成功率）：π0.5 0.20，+规划器 0.34，+VGGT-Ω 0.40，完整 0.46。
- 训练：批大小 48；ACT/DP 40k 步；π0.5/HMAP 按任务 20k 或 30k 步；重规划间隔 40 控制步（部分任务）。附录自述优化器、LoRA、checkpoint 选择等待补充。
- 局限（论文 §6）：以仿真为主，真机仅 1 任务 × 10 次；五类记忆不穷尽；只有成功专家示范；阶段指标无法完全区分记忆错误与执行错误；SugarCount 各阶段适用 episode 子集不同。

## 对 wiki 的映射

- [paper-robocasa-mem](../../wiki/entities/paper-robocasa-mem.md)
- [按路段调速的规划、虚拟房间到真机的人形动作与人类视频缩放：具身智能小站 11-06 期 20 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-06-20-papers-technology-map.md)
