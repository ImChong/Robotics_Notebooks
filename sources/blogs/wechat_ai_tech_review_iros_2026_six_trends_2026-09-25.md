# 研究了1933篇 IROS 论文，我们看到了机器人学的六项新变化

> 来源归档（blog / 微信公众号）

- **标题：** 研究了1933篇 IROS 论文，我们看到了机器人学的六项新变化
- **类型：** blog
- **作者：** AI科技评论（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/XvdbbidbJKBszMQwVzvD0A
- **发表日期：** 2026-09-25
- **入库日期：** 2026-09-25
- **抓取方式：** Camoufox + wechat-article-for-ai（`sources/raw/wechat_ai_tech_review_iros_2026_six_trends_2026-09-25/article.md`）
- **一句话说明：** 基于 IROS 2026 正式议程 **1933 篇**多标签统计的策展解读：AI 嵌入规划/几何/控制/触觉而非取代传统机器人学；VLA 进入补短板与系统工程阶段；Reasoning/Memory 中间层复兴；Manipulation 与 loco-manipulation 仍是主战场；World Model 仅 **19 篇（~1%）** 但向控制参与过渡。

## 核心摘录（归纳，非全文）

### 统计口径（文内多标签，不可简单相加）

| 标签 | 约篇数 | 备注 |
|------|--------|------|
| Robot Learning / Embodied AI | 809 | 与 Manipulation 交叉 ~276 |
| Navigation / Planning | 564 | 与 Learning 交叉 ~225 |
| Perception / Vision | 556 | 与 Learning 交叉 ~269 |
| Control / Dynamics | 546 | 与 Learning 交叉 ~221 |
| Manipulation | 520 | Dexterous/Tactile 子集 ~225 |
| Humanoid / Legged | 213 | Loco-manipulation 子集 ~89 |
| VLA/VLM/LLM/Foundation | 162（8.4%） | 明确 VLA ~82 |
| Reasoning / Memory | 119（6.2%） | Reasoning ~64，Memory ~36 |
| World Model | 19（~1%） | 热但量小 |

### 六项变化（对应 overview 六节）

1. **AI 重新嵌入传统栈** — LLM+PDDLStream TAMP、3D 几何补 VLA、慢 FM 训快策略、视触觉 Foundation Model。
2. **VLA 从「证明能做」到补短板** — 蒸馏提速、3D/视角/长时记忆、RoboBRIDGE 式系统壳。
3. **中间层复兴** — Temporal KV、3D Gaussian Memory、Dual CoT、神经符号 VLN。
4. **Manipulation 密集战场** — VTAP/PDS 硬件、TacVLA vs HapticVLA、钢琴指法数据。
5. **Humanoid loco-manipulation** — ULTRA、SteadyTray、DreamMimic、专家路由 VLA、网球技能。
6. **World Model 向控制参与过渡** — DAWN 深度去噪、跨本体 WM、插接 WM、RoDyn 2.5D。

## 文内代表论文 → 本库节点

> 本文为 **趋势解读**，非「一篇论文一个 ingest」盘点；已入库节点 **复用**，高信号且库内缺失者 **新建实体**（见 [catalog](../papers/iros_2026_ai_tech_review_six_trends_cited_papers_catalog.md)）。

| 代表工作 | 本库节点 | 动作 |
|----------|----------|------|
| GeoVLA | [paper-geovla](../../wiki/entities/paper-geovla.md) | **新建** |
| Shallow-π | [paper-shallow-pi](../../wiki/entities/paper-shallow-pi.md) | **新建** |
| AnyCamVLA | [paper-anycam-vla](../../wiki/entities/paper-anycam-vla.md) | 复用 |
| VTAP Gripper | [paper-vtap-gripper](../../wiki/entities/paper-vtap-gripper.md) | 复用 |
| HapticVLA | [paper-sa-2603-15257-hapticvla…](../../wiki/entities/paper-sa-2603-15257-hapticvla-contact-rich-manipulation-via-vision-l.md) | 复用（清单索引） |
| ULTRA | [paper-notebook-ultra…](../../wiki/entities/paper-notebook-ultra-unified-multimodal-control-for-autonomous.md) | 复用 |
| SteadyTray | [paper-notebook-steadytray](../../wiki/entities/paper-notebook-steadytray.md) | 复用 |
| DreamMimic | [paper-dreammimic](../../wiki/entities/paper-dreammimic.md) | 复用 |
| LATENT（人形网球） | [paper-notebook-latent](../../wiki/entities/paper-notebook-latent.md) | 复用 |
| 其余文内举例 | [catalog](../papers/iros_2026_ai_tech_review_six_trends_cited_papers_catalog.md) | 索引待后续单篇 ingest |

## 对 wiki 的映射

- 阅读坐标：[IROS 2026 六趋势技术地图](../../wiki/overview/iros-2026-six-trends-technology-map.md)
- 交叉：[VLA](../../wiki/methods/vla.md)、[World Action Models](../../wiki/concepts/world-action-models.md)、[loco-manipulation](../../wiki/tasks/loco-manipulation.md)

## 当前提炼状态

- [x] 公众号正文抓取
- [x] GeoVLA / Shallow-π 步骤 2.5 与实体升格
- [x] 六趋势 overview + 代表论文 catalog
