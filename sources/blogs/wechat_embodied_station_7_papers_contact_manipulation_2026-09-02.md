# 7 个开源论文一次看完：具身智能机器人最新论文速览

> 来源归档（blog / 微信公众号）

- **标题：** 7个开源论文一次看完：具身智能机器人最新论文速览
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/v2-G3TNZV5e_Uzm0kHPZEA
- **发表日期：** 2026-09-02
- **入库日期：** 2026-09-02
- **抓取方式：** Agent Reach + `wechat-article-for-ai`（Camoufox）；`--no-images`
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_7_papers_contact_manipulation_2026-09-02.md`](../raw/wechat_embodied_station_7_papers_contact_manipulation_2026-09-02.md)
- **一句话说明：** 汇总 7 篇近期机器人与具身开源论文，主线为接触丰富精密装配、模仿学习时间鲁棒性、神经符号运动规划、单目深度综述、工业点云配准、高精度插入 benchmark 与非抓取投掷；**7/7 均有独立 `paper-*` 详情节点**（本 ingest **新建 7**；同一 arXiv **不重复造页**）。

## 核心摘录（归纳，非全文）

文内判断：具身智能正从「单一大模型策略」走向更工程化的开源闭环——接触力、时间尺度、符号推理、几何感知、基准硬件与低成本传感器，重新进入机器人系统设计核心。

### 7 篇 → 本库节点

| # | 论文 | arXiv | 开源结论（入库日） | wiki |
|---|------|-------|-------------------|------|
| 01 | Facet-0 | [2609.01596](https://arxiv.org/abs/2609.01596) | **已开源** `PINE-Lab-NTU/FACET` + HF 模型/数据集 | [paper-facet-0](../../wiki/entities/paper-facet-0.md) |
| 02 | ParcelStow | [2609.01453](https://arxiv.org/abs/2609.01453) | **已开源** `coenwerem/parcelstow` | [paper-parcelstow](../../wiki/entities/paper-parcelstow.md) |
| 03 | Dual Process Motion Planning | [2609.01260](https://arxiv.org/abs/2609.01260) | **已开源** `verayannn/System-1-and-System-2-in-Motion-Planning` | [paper-dual-process-motion-planning](../../wiki/entities/paper-dual-process-motion-planning.md) |
| 04 | Monocular Depth Survey | [2609.01172](https://arxiv.org/abs/2609.01172) | **已开源** `CVMI-Lab/Depth_Survey` | [paper-monocular-depth-estimation-survey](../../wiki/entities/paper-monocular-depth-estimation-survey.md) |
| 05 | ADM-BA | [2609.01089](https://arxiv.org/abs/2609.01089) | **已开源** `YiranZhou-Robotics/ADM-BA` | [paper-adm-ba](../../wiki/entities/paper-adm-ba.md) |
| 06 | Peg-in-Bench | [2609.00906](https://arxiv.org/abs/2609.00906) | **待核实**：论文声明 `aistairc/peg-in-bench`，截至入库日仓库 404 | [paper-peg-in-bench](../../wiki/entities/paper-peg-in-bench.md) |
| 07 | NP-Throw | [2609.00771](https://arxiv.org/abs/2609.00771) | **已开源** `Abdullah-AIST/NP-Throw` | [paper-np-throw](../../wiki/entities/paper-np-throw.md) |

### 文内要点速记

1. **Facet-0** — action-wrench 联合建模 + flow matching；ManuFacet-1K 训练；5 个亚毫米装配任务 82% 成功率。
2. **ParcelStow** — 标称速度下专家与 ACT 均 100%，最高示范速度处专家 84% / ACT 53%；插入错位为主失败模式。
3. **Dual Process MP** — System-1 经验策略 + System-2 符号求解器，元认知控制器动态调度。
4. **Depth Survey** — 单目深度从经典学习到基础模型时代的判别式/生成式范式梳理。
5. **ADM-BA** — 无对应点云配准；分层深度图 BA 用于废钢工业多视角重建。
6. **Peg-in-Bench** — 可 3D 打印模块化插孔基准，支持公差/几何/布局组合泛化评测。
7. **NP-Throw** — RL 优化关节轨迹，利用滑动/滚动接触；UR5e 零样本 97% 真机成功率。

## 对 wiki 的映射

- **7/7 独立详情节点**：每篇对应唯一 `wiki/entities/paper-*.md`；静态站 `detail.html?id=entity-paper-…` 均可直达。
- **本 ingest 新建 7** 个实体；**0 重复 arXiv 节点**。
- 阅读坐标：[接触丰富操作 7 篇技术地图](../../wiki/overview/contact-rich-manipulation-7-papers-technology-map.md)（**非**论文详情替代，仅作横切面索引）。
- 交叉：[Manipulation](../../wiki/tasks/manipulation.md)、[Imitation Learning](../../wiki/methods/imitation-learning.md)、[Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 7 篇独立节点核查（7 新建 / **0 重复 arXiv 节点**）
- [x] 项目页与仓库开源状态核查（步骤 2.5）
