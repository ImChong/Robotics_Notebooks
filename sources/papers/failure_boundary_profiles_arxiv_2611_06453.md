# Failure-Boundary Profiles（arXiv:2611.06453）

> 来源归档（paper）

- **标题：** Beyond Infraction Scores: Failure-Boundary Profiles for Closed-Loop Evaluation of Driving and Robot Navigation Agents
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.06453>
- **PDF：** <https://arxiv.org/pdf/2611.06453>
- **代码：** <https://github.com/speedym/failure-boundary-profiles/tree/v1.0-paper>
- **入库日期：** 2026-11-06
- **线索来源：** [只在虚拟房间练习，人形机器人走进陌生房间能做什么？](../blogs/wechat_embodied_station_20_papers_curobov2_fast_halo_humanwam_2026-11-06.md)（具身智能小站，2026-11-06）
- **通读：** 已完成
- **核对范围：** 已下载 arXiv PDF 全文并通读摘要、引言、相关工作、方法（指标 / 认证管线）、七个 CARLA 族与三个仓库族的实验、训练前后重测、局限与结论；未逐条核对参考文献，未运行代码，图中曲线只读标题与图注数值。
- **一句话说明：** 逐步收窄通道，分别记录策略尝试通过和无碰撞通过的概率。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-06）：资源入口 speedym/failure-boundary-profiles（v1.0-paper 标签）经 git ls-remote 可达；无项目页。

## 核心摘录（arXiv v1 全文核对，数字出自论文）

- **作者 / 版本：** Bulat、Ardelean、Kirby（Social Chaos Lab / valeo.ai），arXiv v1，2026-11-04，cs.RO。
- **问题：** CARLA 驾驶分（路线完成度 × 违规惩罚）分不清「停在障碍前」和「挤过去」。Charger 堵路走廊的阻塞宽度上，擦碰通过的 SimLingo 均分 52.2，全部停车的智能体为 37.8 至 40.6（BridgeDrive 36.1，不停车且低分）。
- **方法：** 每个场景族只扫一个实测因子（走廊自由宽度或对向车流时间间隙），每个变体按测量结果认证为应通过（passable）或应停（blocked/wait）。估计两条曲线：commitment（尝试通过概率 P_go）与 competence（无接触通过概率 P_clean），取 p=0.5 处为失败边界（逻辑回归拟合，区间为按变体重采样的 bootstrap，无法拟合时给相邻变体 bracket）。run 间非确定（同变体同种子五次重复得 53.7 与 32.2），故每次 run 视作独立抽样并按变体合并种子。
- **决策准确率（DA）：** 应通过变体上的无接触通过率与应停变体上的停车率的平衡准确率；总是停得 0.5，总是走至多 0.5。仅在含停车侧的 3 个族上等权平均。
- **场景生成管线（SCL）：** 自然语言意图 → 规划器生成声明式计划 → 确定性求解器与验证器 → 变体在模拟器内回读核对（位姿漂移至多 10 mm）→ 跑 campaign → 认证。对向车流用间隙证书（可用 / 条件 / 等待，0.5 s 余量）。仓库含 10 个场景族与 17 次 campaign 记录。
- **规模：** 7 个 CARLA 智能体 x 7 族（每变体 3 个种子）；仓库货运机器人 X-Mobility 在 Isaac Sim 3 个仓库族（每间隙 5 次策略采样，Isaac Sim 按位复现）。
- **主结果（Table 2，DA / 均分 DS）：** 专家 0.74 / 79.8；TFv6 0.60 / 75.5（学习型最高）；PlanT 2.0 ckpt1 0.56；TF++ 0.49；HiP-AD 0.44；SimLingo 0.28；BridgeDrive 0.26（DS 71.8 排第四，却从不停车）；X-Mobility 0.34。
- **失败边界例：** Charger 走廊 TFv6 commitment 边界 2.82 m、competence 2.84 m；BridgeDrive 与 SimLingo 全宽度都尝试，分别 2.90 m 与 3.38 m 以上才干净通过；政策下限 2.30 m，车身 1.84 m。
- **对向车流：** 29 个等待窗口中专家等 13 次、TFv6 6 次、BridgeDrive 3 次，其余学习型 0 次。
- **仓库机器人微调：** 全模型微调使可通过间隙全部干净通过，但阻塞间隙 stop 从 15/15 降为 0/15（留出的 composed stacks），平均 DA 0.34 升至 0.50；仅头部微调 DA 降至 0.22。机器人从不绕行相邻通道。
- **稳健性：** 把 0 至 1 s 的证书余量及 2.07 m 宽度改判为可通过，排名两端不变（专家 0.70 至 0.83、TFv6 0.58 至 0.64、SimLingo 与 BridgeDrive 0.22 至 0.33），中间组换序。
- **局限（论文自述）：** 全部为仿真，不知真车 / 真机是否同样；标签依赖声明的常量（政策下限、证书余量）；汇总仅基于 3 个含停车侧的族；专家是脚本规划器；等待窗口是规则而非物理不可能；证书只能跑完 campaign 后给出；一族中存在场景缺陷（如最短间距 0.45 s 变体、0.99 m 贩卖机倒下）被排除；仿真平台 Isaac Sim 5.1 单一版本，不跨版本合并。
- **数据 / 代码：** 仓库 tag v1.0-paper；模拟输出在 Hugging Face（DOI 10.57967/hf/11195）；仓库不含仓库运行器、CARLA 地图和各智能体检查点。

## 对 wiki 的映射

- [paper-failure-boundary-profiles](../../wiki/entities/paper-failure-boundary-profiles.md)
- [按路段调速的规划、虚拟房间到真机的人形动作与人类视频缩放：具身智能小站 11-06 期 20 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-06-20-papers-technology-map.md)
