# EgoMatrix: Physically Faithful Dexterous Robot Demonstrations from Egocentric Human Videos（arXiv:2610.18822）

> 来源归档（ingest）

- **标题：** EgoMatrix: Physically Faithful Dexterous Robot Demonstrations from Egocentric Human Videos
- **arXiv：** <https://arxiv.org/abs/2610.18822>（HTML：<https://arxiv.org/html/2610.18822v1>）
- **代码：** 未列（见「开源状态」）
- **项目页：** <https://spikew726.github.io/EgoMatrix/>（归档：[egomatrix-project.md](../sites/egomatrix-project.md)）
- **作者：** Ziheng Wang、Yanming Shao、Ting Mao、Kaihan Chen、Yiqun Wang、Xuanye Wu、Yao Mu；机构：上海交通大学、香港大学、浙江大学、上海人工智能实验室
- **入库日期：** 2026-10-17
- **来源线索：** 具身智能小站 2026-10-17 文章（<https://mp.weixin.qq.com/s/I4XP5JSG9QNmm4hd1nk2JQ>）
- **一句话说明：** 单目第一视角视频到可回放双手灵巧轨迹与配对机器人观测的数据引擎。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-17）：** 仅 arXiv 链接与视频；无 GitHub / 数据集 / coming soon。
- **论文核查：** 无代码发布声明，仅链接项目页。
- **结论：** **未列链接**：截至 2026-10-17 无公开代码或数据。

## 摘录 1：管线

- 三阶段：接触感知重建（手锚定初始化、多起点精修、局部接触精修）；灵巧操作迁移（关键点重定向 + 联合双手 PPO，Isaac Lab）；演示合成与背景迁移（渲染、仿真阴影、单次刚体变换换背景）。

**对 wiki 的映射：** [`paper-egomatrix.md`](../../wiki/entities/paper-egomatrix.md)「核心原理」。

## 摘录 2：结果

- 62 条演示：54.84%（34/62）对 Do As I Do 14.52%、DexImit 12.90%；TACO 消融 5.41% → 27.03% → 48.65%。
- Fréchet 距离 5,252 帧：ResNet18 100.23、VGG16 414.66、DreamSim 0.4975；DINOv2 上 GPT-6 Astra 更好。

**对 wiki 的映射：** 「实验与评测」「结论」。

## 摘录 3：局限

- 管线开销、单目深度歧义、下游策略训练未验证。

**对 wiki 的映射：** 「局限与风险」。
