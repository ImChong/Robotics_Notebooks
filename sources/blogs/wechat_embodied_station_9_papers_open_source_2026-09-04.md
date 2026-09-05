# 9篇具身智能新论文开源！VLA、双臂家务、人形机器人、失败检测全来了

> 来源归档（blog / 微信公众号）

- **标题：** 9篇具身智能新论文开源！VLA、双臂家务、人形机器人、失败检测全来了
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/IDeWoG3ykIlyPJJcYpPLhg
- **发表日期：** 2026-09-04
- **入库日期：** 2026-09-04
- **抓取方式：** headless Chrome + iPhone UA（Jina / 桌面 UA 返回微信「环境异常」验证页）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_9_papers_open_source_2026-09-04.md`](../raw/wechat_embodied_station_9_papers_open_source_2026-09-04.md)
- **一句话说明：** 2026 年开源系列续期，汇总 9 篇近期机器人与具身论文，主线为动作足够用的中间表征、跨手型视觉语言抓取、工业对话数据、可复现网络装配、LIBERO 容量下限、VLM 失败检测、1500 小时双臂家务、开源人形共设计与拆解夹爪；**9/9 均有独立 `paper-*` 详情节点**（本 ingest **新建 9**；同一 arXiv **不重复造页**）。

## 核心摘录（归纳，非全文）

文内判断：具身智能正从单点 demo 进入「数据、代码、检查点、硬件材料、评测框架一起开源」的可复现阶段；竞争点是控制相关表征、可泛化操作、可靠失败识别与低成本开放硬件。

### 9 篇 → 本库节点

| # | 论文 | arXiv | 开源结论（入库日） | wiki |
|---|------|-------|-------------------|------|
| 01 | GIFT | [2609.04193](https://arxiv.org/abs/2609.04193) | **待发布**：仅项目页 `GIFT-pages`，无训练/推理仓 | [paper-gift-intermediate-feature-training](../../wiki/entities/paper-gift-intermediate-feature-training.md) |
| 02 | AdaRoboVLG | [2609.04096](https://arxiv.org/abs/2609.04096) | **待发布**：仅项目页，未见 GitHub | [paper-adarobovlg](../../wiki/entities/paper-adarobovlg.md) |
| 03 | IRWOZ 2.0 | [2609.04030](https://arxiv.org/abs/2609.04030) | **部分开源**：IEEE Dataport 数据集 + 旧仓 `lcroy/ToD4IR`（2022） | [paper-irwoz-2](../../wiki/entities/paper-irwoz-2.md) |
| 04 | Network Design | [2609.03852](https://arxiv.org/abs/2609.03852) | **已开源** `Barabasi-Lab/NetworkDesign`（`UnigraphCheck` + Data） | [paper-network-design-reproducible](../../wiki/entities/paper-network-design-reproducible.md) |
| 05 | MINERVA | [2609.03715](https://arxiv.org/abs/2609.03715) | **已开源** `k1000dai/MINERVA`（Apache-2.0）+ HF 权重 | [paper-minerva-libero](../../wiki/entities/paper-minerva-libero.md) |
| 06 | FailBench | [2609.03611](https://arxiv.org/abs/2609.03611) | **部分开源**：`Metric-AI-Lab/failbench` 仅站点镜像，harness 按钮仍为 `#` | [paper-failbench](../../wiki/entities/paper-failbench.md) |
| 07 | XR-2 / 双臂家务 | [2609.03591](https://arxiv.org/abs/2609.03591) | **部分开源**：HF `challenge-2026/challenge_data`（CC-BY-SA-4.0）；策略代码未见 | [paper-xr2-bimanual-household](../../wiki/entities/paper-xr2-bimanual-household.md) |
| 08 | BRIDGE | [2609.03497](https://arxiv.org/abs/2609.03497) | **部分开源**（2026-09-05 再核）：项目页 `.stp` CAD；控制/BOM 待录用 | [paper-bridge-humanoid](../../wiki/entities/paper-bridge-humanoid.md) |
| 09 | ARTiS | [2609.03362](https://arxiv.org/abs/2609.03362) | **部分开源**：项目页 CAD/BOM/hardware；仓为站点镜像 | [paper-artis-gripper](../../wiki/entities/paper-artis-gripper.md) |

### 文内要点速记

1. **GIFT** — 几何 / 可供性 / 目标区域监督中间特征；LIBERO-Plus 上 GIFT-VLA / Fast / IDM 达 79.6% / 72.6% / 87.8%。
2. **AdaRoboVLG** — 物理抓取生成与任务语义解耦；跨手型 force-closure 候选 + 可组合基础模型先验。
3. **IRWOZ 2.0** — 390 段工业对话、4 域；GPT-2 BLEU-4 0.1651→0.5604。
4. **Network Design** — Unigraphical Design Theorem + guided assembly；3618 个可复现系统 + 3D 打印验证。
5. **MINERVA** — 0.54M 参数 LIBERO 平均约 95%；容量在 ~1M 饱和、&lt;0.25M 崩塌。
6. **FailBench** — 2197 次尝试、13 个 VLM 检测器，最好 balanced accuracy 仅 0.77；接触装配接近随机。
7. **XR-2** — 开放 1500 小时双臂家务示范；示范量与 DAgger 修正均呈稳定 scaling。
8. **BRIDGE** — 88 cm、约 1500 美元开源人形；形态–控制共设计，对照 Bumi / K1 / Toddlerbot。
9. **ARTiS** — jamming 掌 + fin-ray 指尖的三指拆解夹爪（TASE）。

## 对 wiki 的映射

- **9/9 独立详情节点**：每篇对应唯一 `wiki/entities/paper-*.md`；静态站 `detail.html?id=entity-paper-…` 均可直达。
- **本 ingest 新建 9** 个实体；**0 复用 / 0 重复 arXiv 节点**。
- 阅读坐标：[开源可复现性 9 篇技术地图](../../wiki/overview/open-source-reproducibility-9-papers-technology-map.md)（**非**论文详情替代，仅作横切面索引）。
- 交叉：[VLA](../../wiki/methods/vla.md)、[World Action Models](../../wiki/concepts/world-action-models.md)、[Manipulation](../../wiki/tasks/manipulation.md)、[Humanoid Locomotion](../../wiki/tasks/humanoid-locomotion.md)。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 9 篇独立节点核查（9 新建 / **0 重复 arXiv 节点**）
- [x] 项目页与仓库开源状态核查（步骤 2.5）
