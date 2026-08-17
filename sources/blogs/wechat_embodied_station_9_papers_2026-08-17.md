# 机器人论文又卷到哪了？9篇新作看懂具身智能的下一步

> 来源归档（blog / 微信公众号）

- **标题：** 机器人论文又卷到哪了？9篇新作看懂具身智能的下一步
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/UsgswMgDw4Kdpt5qI9fxnA
- **发表日期：** 2026-08-17
- **入库日期：** 2026-08-17
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + `wechat-article-for-ai`（Camoufox）；`--no-images`；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_9_papers_2026-08-17.md`](../raw/wechat_embodied_station_9_papers_2026-08-17.md)
- **一句话说明：** 汇总 9 篇近期具身/机器人论文（文内均给项目页或代码链），主线从「堆大模型」转向把 **未来预测、语言语义、空间轨迹、速度/安全成本** 接到控制闭环；本库 **9 篇均无既有 complete 节点，全部新建独立论文实体，不重复造页**。**注意：** 文内 SG-WAM（语义引导，arXiv:2608.08839）与另一篇同缩写 *Self-Guided World Modeling*（arXiv:2608.01397）**不是同一工作**。

## 核心摘录（归纳，非全文）

文内判断：WAM 正在从「生成好看的未来」转向「生成对动作有用的未来」；SpeedTuning、PGIF-MPPI、PEEL、SurgLAT 则强调执行端的速度、安全和几何约束。落地关键不是模型规模，而是世界模型、VLM 语义、MPC/采样规划与真机部署之间的接口是否干净。

### 9 篇 → 本库节点

| # | 论文 | arXiv | 开源结论（入库日） | wiki |
|---|------|-------|-------------------|------|
| 01 | SpeedTuning | [2608.09138](https://arxiv.org/abs/2608.09138) | **已开源** 仿真复现仓（MIT） | [paper-speedtuning](../../wiki/entities/paper-speedtuning.md) |
| 02 | SHRIMP | [2608.08884](https://arxiv.org/abs/2608.08884) | **已开源** Docker + Isaac Sim + Franka 栈 | [paper-shrimp](../../wiki/entities/paper-shrimp.md) |
| 03 | SG-WAM（语义引导） | [2608.08839](https://arxiv.org/abs/2608.08839) | 项目页 404；**实现未开源** | [paper-sg-wam-semantic-guidance](../../wiki/entities/paper-sg-wam-semantic-guidance.md) |
| 04 | LAMDA | [2608.08815](https://arxiv.org/abs/2608.08815) | 论文给 GitHub，仓 **404** | [paper-lamda-tsr](../../wiki/entities/paper-lamda-tsr.md) |
| 05 | PEEL | [2608.08773](https://arxiv.org/abs/2608.08773) | 双盲匿名仓 **anonymous.4open.science** | [paper-peel-disassembly](../../wiki/entities/paper-peel-disassembly.md) |
| 06 | PGIF-MPPI | [2608.08323](https://arxiv.org/abs/2608.08323) | **已开源** JAX 仿真评测（MIT） | [paper-pgif-mppi](../../wiki/entities/paper-pgif-mppi.md) |
| 07 | 4D-WAM | [2608.08023](https://arxiv.org/abs/2608.08023) | **已开源** FastWAM / Lingbot-VA 后训练 | [paper-4d-wam](../../wiki/entities/paper-4d-wam.md) |
| 08 | SurgLAT | [2608.07876](https://arxiv.org/abs/2608.07876) | 项目页已发；独立训练仓未找到 | [paper-surglat](../../wiki/entities/paper-surglat.md) |
| 09 | V-Simba | [2608.07870](https://arxiv.org/abs/2608.07870) | **已开源** 视觉 SAC 架构（Apache-2.0） | [paper-v-simba](../../wiki/entities/paper-v-simba.md) |

### 文内要点速记

1. **SpeedTuning** — 冻结模仿策略，只学离散速度倍率；倒/抛/取上超过 2.4× 加速。
2. **SHRIMP** — 自然语言 → 层级 primitive，仿真里反复改，再上真机；N=35 提升控制感与透明度。
3. **SG-WAM（语义）** — VLM 出 text-grounded + spatial-aware foresight，注入 WAM；LIBERO 98.7%。**勿与 Self-Guided SG-WAM 合并。**
4. **LAMDA** — 训练期把 VLM 语言原型蒸馏进交通标志识别，推理零负担；阴影 +12.5 pp。
5. **PEEL** — MAB-RRT + 并行批次赛跑求拆解顺序；76 装配体 100%，Fetch 真机 10–17 件。
6. **PGIF-MPPI** — 行人预测写成各向异性高斯场；300 场景碰撞率 0%，Hard 超时 59%。
7. **4D-WAM** — 轨迹场 motion / destination alignment；LIBERO-Plus +8.8 pp。
8. **SurgLAT** — 隐式手术注意力 + RCM 约束腹腔镜控制；SZPH IoU 0.604。
9. **V-Simba** — 把 Simba 归一化/点卷积接到视觉 SAC；DMC / Adroit / Meta-World。

## 对 wiki 的映射

- **新建 9** 个独立 `paper-*` 详情节点；各页 `参考来源` 回链本博客。
- 交叉：[World Action Models](../../wiki/concepts/world-action-models.md)、[MECo-WAM](../../wiki/entities/paper-meco-wam-4d-geometry-cotraining.md)、[MPPI](../../wiki/methods/mppi.md)、[SAC](../../wiki/methods/sac.md)、[模仿学习](../../wiki/methods/imitation-learning.md)、[VLA](../../wiki/methods/vla.md)、[LIBERO](../../wiki/entities/libero-benchmark.md)、[4D 几何分类](../../wiki/overview/wm-action-consequence-category-03-geometry-4d.md)。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 9 篇独立节点核查（0 复用 / 9 新建 / 0 stub 重复）
- [x] 项目页与仓库开源状态核查（步骤 2.5）
