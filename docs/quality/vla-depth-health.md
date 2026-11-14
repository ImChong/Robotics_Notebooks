# VLA 纵深路线节点健康度补齐记录（2026-11-14）

将 [`roadmap/depth-vla.md`](../../roadmap/depth-vla.md)（`roadmap.html?id=roadmap-depth-vla`）Stage 0–5 全量 **80** 个去重 `wiki/` 知识节点（连同 **6** 条 Stage 关联子路线 `roadmap/depth-*.md`、地图渲染共 **86** 个主线节点）与路线页自身统一补齐至 **`3/3`** 满分健康度。

## 总览对比（相对 `origin/main` 基线 `74898814d`）

| 指标 | `origin/main` 基线 (`74898814d`) | 本次完成后 |
|------|-------------------------------|------------|
| `roadmap/depth-vla.md` 自身健康度 | `3/3`（`gaps=0`，91 条内链） | `3/3`（`gaps=0`，91 条内链） |
| Stage 0–5 去重 `wiki/` 节点总数 | `80` | `80` |
| Stage `3/3`（满分） | `26` | **`80`** |
| Stage `2/3` | `19` | `0` |
| Stage `1/3` | `35` | `0` |
| Stage `0/3` | `0` | `0` |
| Stage 关联 `roadmap/depth-*.md`（`3/3`） | `6 / 6` | `6 / 6` |
| 页脚非 Stage 引用（范围外） | `1`（[`glancewam-vla-crew-10-papers-technology-map.md`](../../wiki/overview/glancewam-vla-crew-10-papers-technology-map.md)，`1/3`） | `1`（按 [`audit_depth_health.py`](../../scripts/audit_depth_health.py) 与 [WAM 治理记录](./wam-depth-health.md) 惯例保持范围外，不伪造未通读论文的 `通读：已完成`） |

本次 Stage `54` 个低于满分节点中：**52** 篇直接在 `wiki/` 补全结构化九节骨架、Mermaid 流程图/源码运行时序图、`## 英文缩写速查` 与可核对的一手来源；另有 **2** 篇（[`paper-eventvla-visual-evidence-memory.md`](../../wiki/entities/paper-eventvla-visual-evidence-memory.md) 与 [`paper-robo-harness.md`](../../wiki/entities/paper-robo-harness.md)）因补全其 `original:` 对应来源文件的 `通读：已完成` 而直接升格为 `3/3`。同期新建 `3` 篇 arXiv 来源归档与 `5` 篇仓库源码归档，修订 `33` 篇论文、`20` 篇项目站点、`23` 篇代码仓与 `1` 篇数据集归档，并同步校准 [`roadmap/depth-vla.md`](../../roadmap/depth-vla.md) Stage 2–5 中的模型规模、开源边界与撤稿状态描述。

## 补齐方式与核对边界

1. **全文通读与源码入口核对：** 为每篇缺口节点归档本地离线原文（arXiv PDF 文本 / HTML、`github.io` 静态页面、Hugging Face Hub API `manifest.json`、官方 GitHub 代码树）。每篇有可运行实现的工作均逐文件核查 README、CLI 入口与关键模块并补写真实 `## 源码运行时序图`；无公开可运行实现时在 `## 源码运行时序图` 显式写明「不适用」及具体原因。
2. **已知边界如实标注（不臆造、不越界）：**
   - **RTCF (`arXiv:2608.04527`) v3 撤稿：** 一作 Yuze Fan 已于 `2026-10-28` 提交 `v3` 撤稿声明；wiki 与路线页明确标为「v3 已撤稿，基于 v2 全文作方法脉络归档、无公开代码」，不在跨页引用中误标为已验证定论。
   - **TEMPO (`arXiv:2608.07314`) 匿名页 HTTP 404：** 原文附带 `<https://anonymous.4open.science/w/tempo-page/>` 访问返回 `HTTP 404 Not Found`（见 [`tempo-anonymous-4open.md`](../../sources/sites/tempo-anonymous-4open.md)），归档为未开源并区分于同期同名工作 `tempo-robot/TEMPO` (`arXiv:2609.16864`)。
   - **vla.simd (`arXiv:2609.24274`) 双盲匿名代码：** 项目页仅挂出 `anonymous.4open.science/r/vla-simd` 与 `lerobot-vla-simd`，页脚声明正式代码与权重待论文接收后公布；归档中如实注明尚未发布实名 GitHub 仓库，未编写源码运行时序图。
   - **ReflexVLA (`arXiv:2608.14379`) 仅开源基准与数据：** `LxRoboticsLab/ReflexBench`（BSD 3-Clause）与 `cyx337/ReflexBench_dataset` 已公开，但逐文件检索确认仓库不含 ReflexVLA 策略训练/推理源码（项目页仍标 `Code After acceptance`），按 `code_scope` 严格限定时序图范围。
   - **FlashVLA 与 Galaxea G0.5 站点快照边界：** `sources/sites/z-lab-flashvla.md`（未归档本地站点快照）保持 `通读：未完成`，`sources/sites/opengalaxea-g05.md` 抓取结果为 458 B Vite/React SPA 空壳亦未计入通读完成；两者均仅以已完成全量通读的 arXiv 全文与官方 GitHub 仓库作为 `original:` 一手依据。
3. **页脚多论文综述外推排除：** `roadmap/depth-vla.md` 末尾拓展阅读中的 [`glancewam-vla-crew-10-papers-technology-map.md`](../../wiki/overview/glancewam-vla-crew-10-papers-technology-map.md)（`stage=False`，`1/3`）关联 10 篇论文来源（当前仅 2 篇标为通读完成），依仓库既有策略保留为范围外页面。

## 共享来源的反向引用复核

本次更新的 `36` 篇 `sources/papers/*.md` 共反向关联 `91` 个 `wiki/` 页面，使用 [`scripts/bump_wiki_updated_for_sources.py`](../../scripts/bump_wiki_updated_for_sources.py) 同步 `updated: 2026-11-14`，避免来源新于知识页的陈旧告警（0 条新增陈旧告警）。其中 Stage 内已为 `3/3` 的 **6** 个页面同步刷新日期，Stage 外 **45** 个共享引用页面（当前健康度分布：`3/3` 19 篇、`2/3` 8 篇、`1/3` 18 篇）除按最新开源/撤稿事实微调 [`manipulation.md`](../../wiki/tasks/manipulation.md)、[`vla-deployment-guide.md`](../../wiki/queries/vla-deployment-guide.md)、[`embodied-station-2026-10-18-26-papers-technology-map.md`](../../wiki/overview/embodied-station-2026-10-18-26-papers-technology-map.md)、[`rekadaily-10k-dataset.md`](../../wiki/entities/rekadaily-10k-dataset.md)、[`paper-dreamsteer-vla-deployment-steering.md`](../../wiki/entities/paper-dreamsteer-vla-deployment-steering.md) 与 [`paper-dynawm-vla-online-correction.md`](../../wiki/entities/paper-dynawm-vla-online-correction.md) 中的对应陈述外，其余低于 `3/3` 的范围外节点保留原 `original:` 与 `status:`，留待所属纵深路线批次补齐。

## Stage 0–5 节点明细（80/80 满分）

### Stage 0 具身基础模型全景与前置（8 条引用）

| 节点 | 基线 | 基线缺口 | 当前 | 备注 |
|------|------|----------|------|------|
| [`transformer.md`](../../wiki/concepts/transformer.md) | `3/3` | — | **`3/3`** | 基线已满分 |
| [`lerobot.md`](../../wiki/entities/lerobot.md) | `3/3` | — | **`3/3`** | 基线已满分；共享来源反向复核刷新 `updated` |
| [`vlm-vln-vla-vlx-world-model-taxonomy.md`](../../wiki/comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md) | `3/3` | — | **`3/3`** | 基线已满分 |
| [`foundation-policy.md`](../../wiki/concepts/foundation-policy.md) | `3/3` | — | **`3/3`** | 基线已满分 |
| [`hub-embodied-foundation-model.md`](../../wiki/overview/hub-embodied-foundation-model.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`embodied-fm-taxonomy-loop.md`](../../wiki/queries/embodied-fm-taxonomy-loop.md) | `3/3` | — | **`3/3`** | 基线已满分 |
| [`llms-from-scratch-raschka.md`](../../wiki/entities/llms-from-scratch-raschka.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`andrej-karpathy.md`](../../wiki/entities/andrej-karpathy.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |

### Stage 1 模仿学习策略基座（6 条引用）

| 节点 | 基线 | 基线缺口 | 当前 | 备注 |
|------|------|----------|------|------|
| [`behavior-cloning.md`](../../wiki/methods/behavior-cloning.md) | `3/3` | — | **`3/3`** | 基线已满分 |
| [`action-chunking.md`](../../wiki/methods/action-chunking.md) | `3/3` | — | **`3/3`** | 基线已满分；共享来源反向复核刷新 `updated` |
| [`bc-with-transformer.md`](../../wiki/methods/bc-with-transformer.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`diffusion-policy.md`](../../wiki/methods/diffusion-policy.md) | `3/3` | — | **`3/3`** | 基线已满分；共享来源反向复核刷新 `updated` |
| [`diffusion-model.md`](../../wiki/concepts/diffusion-model.md) | `3/3` | — | **`3/3`** | 基线已满分 |
| [`imitation-learning.md`](../../wiki/methods/imitation-learning.md) | `3/3` | — | **`3/3`** | 基线已满分 |

### Stage 2 VLA 主线：从 RT 系列到 π0（26 条引用）

| 节点 | 基线 | 基线缺口 | 当前 | 备注 |
|------|------|----------|------|------|
| [`vla.md`](../../wiki/methods/vla.md) | `3/3` | — | **`3/3`** | 基线已满分 |
| [`hub-vla.md`](../../wiki/overview/hub-vla.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`vla-wm-reading-roadmap-14-papers-technology-map.md`](../../wiki/overview/vla-wm-reading-roadmap-14-papers-technology-map.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`robotics-transformer-rt-series.md`](../../wiki/methods/robotics-transformer-rt-series.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-rt-1.md`](../../wiki/entities/paper-rt-1.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-rt-2.md`](../../wiki/entities/paper-rt-2.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-openvla.md`](../../wiki/entities/paper-openvla.md) | `3/3` | — | **`3/3`** | 基线已满分 |
| [`paper-octo.md`](../../wiki/entities/paper-octo.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-pi0.md`](../../wiki/entities/paper-pi0.md) | `3/3` | — | **`3/3`** | 基线已满分 |
| [`pi07-policy.md`](../../wiki/methods/pi07-policy.md) | `3/3` | — | **`3/3`** | 基线已满分；共享来源反向复核刷新 `updated` |
| [`saycan.md`](../../wiki/methods/saycan.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`dial-instruction-augmentation.md`](../../wiki/methods/dial-instruction-augmentation.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-internvla-a15-unified-vla.md`](../../wiki/entities/paper-internvla-a15-unified-vla.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`lingbot-vla.md`](../../wiki/entities/lingbot-vla.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`kuavo-vla-1.md`](../../wiki/entities/kuavo-vla-1.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`robbyant.md`](../../wiki/entities/robbyant.md) | `3/3` | — | **`3/3`** | 基线已满分 |
| [`paper-bridgevla-plusplus.md`](../../wiki/entities/paper-bridgevla-plusplus.md) | `2/3` | 缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-galaxea-g05.md`](../../wiki/entities/paper-galaxea-g05.md) | `2/3` | 缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-gsr-paravla.md`](../../wiki/entities/paper-gsr-paravla.md) | `2/3` | 缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-indi.md`](../../wiki/entities/paper-indi.md) | `2/3` | 缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-gazevla.md`](../../wiki/entities/paper-gazevla.md) | `3/3` | — | **`3/3`** | 基线已满分 |
| [`paper-topology-informed-visual-prompting-vla.md`](../../wiki/entities/paper-topology-informed-visual-prompting-vla.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-robomp-dinov2.md`](../../wiki/entities/paper-robomp-dinov2.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-x-planner-event-structured-task-planning.md`](../../wiki/entities/paper-x-planner-event-structured-task-planning.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-migu-multimodal-instruction-grounding.md`](../../wiki/entities/paper-migu-multimodal-instruction-grounding.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-tempo-dynamic-manipulation.md`](../../wiki/entities/paper-tempo-dynamic-manipulation.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |

### Stage 3 数据与 Scaling（14 条引用）

| 节点 | 基线 | 基线缺口 | 当前 | 备注 |
|------|------|----------|------|------|
| [`vla-open-source-repro-landscape-2025.md`](../../wiki/overview/vla-open-source-repro-landscape-2025.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-data-pyramid-embodied-manipulation.md`](../../wiki/entities/paper-data-pyramid-embodied-manipulation.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`egoscale.md`](../../wiki/methods/egoscale.md) | `3/3` | — | **`3/3`** | 基线已满分；共享来源反向复核刷新 `updated` |
| [`humannet.md`](../../wiki/entities/humannet.md) | `3/3` | — | **`3/3`** | 基线已满分 |
| [`mimic-video.md`](../../wiki/methods/mimic-video.md) | `3/3` | — | **`3/3`** | 基线已满分 |
| [`world-action-models.md`](../../wiki/concepts/world-action-models.md) | `3/3` | — | **`3/3`** | 基线已满分 |
| [`pelican-unified-1.md`](../../wiki/methods/pelican-unified-1.md) | `3/3` | — | **`3/3`** | 基线已满分 |
| [`defi-decoupled-dynamics-vla.md`](../../wiki/methods/defi-decoupled-dynamics-vla.md) | `3/3` | — | **`3/3`** | 基线已满分 |
| [`embodied-scaling-laws.md`](../../wiki/concepts/embodied-scaling-laws.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`xiaomi-robotics-1.md`](../../wiki/entities/xiaomi-robotics-1.md) | `2/3` | 缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-joyai-ra-05.md`](../../wiki/entities/paper-joyai-ra-05.md) | `2/3` | 缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-ace-data-0.md`](../../wiki/entities/paper-ace-data-0.md) | `2/3` | 缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-roboedit.md`](../../wiki/entities/paper-roboedit.md) | `2/3` | 缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-ego2robot.md`](../../wiki/entities/paper-ego2robot.md) | `3/3` | — | **`3/3`** | 基线已满分；共享来源反向复核刷新 `updated` |

### Stage 4 部署与系统整合（15 条引用）

| 节点 | 基线 | 基线缺口 | 当前 | 备注 |
|------|------|----------|------|------|
| [`xiaomi-robotics-0.md`](../../wiki/entities/xiaomi-robotics-0.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`embodied-fm-latency-generalization-tradeoff.md`](../../wiki/concepts/embodied-fm-latency-generalization-tradeoff.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-evo1-lightweight-vla.md`](../../wiki/entities/paper-evo1-lightweight-vla.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`behavior-tree-vla-orchestration.md`](../../wiki/concepts/behavior-tree-vla-orchestration.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-eventvla-visual-evidence-memory.md`](../../wiki/entities/paper-eventvla-visual-evidence-memory.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补全 `original:` 来源全文通读后升格 |
| [`paper-robottt-test-time-training-vla-context.md`](../../wiki/entities/paper-robottt-test-time-training-vla-context.md) | `3/3` | — | **`3/3`** | 基线已满分 |
| [`paper-actfovea.md`](../../wiki/entities/paper-actfovea.md) | `2/3` | 缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-robo-harness.md`](../../wiki/entities/paper-robo-harness.md) | `2/3` | 缺 original 归档 | **`3/3`** | 补全 `original:` 来源全文通读后升格 |
| [`paper-rtcf.md`](../../wiki/entities/paper-rtcf.md) | `2/3` | 缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-neural-introspection-gating.md`](../../wiki/entities/paper-neural-introspection-gating.md) | `2/3` | 缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-vlaquantbench.md`](../../wiki/entities/paper-vlaquantbench.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-vla-simd.md`](../../wiki/entities/paper-vla-simd.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-reflexvla.md`](../../wiki/entities/paper-reflexvla.md) | `2/3` | 缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-flashvla.md`](../../wiki/entities/paper-flashvla.md) | `2/3` | 缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`manipulation-vla-architecture-selection.md`](../../wiki/queries/manipulation-vla-architecture-selection.md) | `2/3` | 缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |

### Stage 5 进阶方向（12 条引用）

| 节点 | 基线 | 基线缺口 | 当前 | 备注 |
|------|------|----------|------|------|
| [`enpire.md`](../../wiki/methods/enpire.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`safe-real-world-rl-fine-tuning.md`](../../wiki/concepts/safe-real-world-rl-fine-tuning.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-steam-advantage-modeling.md`](../../wiki/entities/paper-steam-advantage-modeling.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-wcm-world-critic-model.md`](../../wiki/entities/paper-wcm-world-critic-model.md) | `2/3` | 缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-tempo.md`](../../wiki/entities/paper-tempo.md) | `2/3` | 缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-autointervene.md`](../../wiki/entities/paper-autointervene.md) | `2/3` | 缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`paper-prism-grpo.md`](../../wiki/entities/paper-prism-grpo.md) | `2/3` | 缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`generative-world-models.md`](../../wiki/methods/generative-world-models.md) | `3/3` | — | **`3/3`** | 基线已满分 |
| [`world-action-models.md`](../../wiki/concepts/world-action-models.md) | `3/3` | — | **`3/3`** | 重复跨阶段引用（见 Stage 3） |
| [`paper-tau0-vla.md`](../../wiki/entities/paper-tau0-vla.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`loco-manip-161-category-09-vla-world-models.md`](../../wiki/overview/loco-manip-161-category-09-vla-world-models.md) | `1/3` | 阅读骨架不完整；缺 original 归档 | **`3/3`** | 补齐骨架/图表/缩写速查与一手来源 |
| [`vision-language-navigation.md`](../../wiki/tasks/vision-language-navigation.md) | `3/3` | — | **`3/3`** | 基线已满分 |

## 复核命令

```bash
python3 scripts/audit_depth_health.py --write
python3 scripts/check_project_nodes.py --base origin/main
GITHUB_ACTIONS=true make ci-preflight
make ci-test
make pr-check
```

## 本地验证结果（2026-11-14）

- `python3 scripts/audit_depth_health.py --write`：`30 routes, 982 unique nodes`；`roadmap/depth-vla.md` 自身 `3/3`，Stage 去重 `wiki/` `80/80` 全部满分（`0` 个 `< 3/3` 节点），`docs/quality/depth-health-backlog.{md,csv}` 已刷新。
- `python3 scripts/check_project_nodes.py --base origin/main` 与 `python3 scripts/pr_derived_guard.py check --base origin/main`：均退出码 `0`（项目身份唯一、未改动任何机器人/部署派生文件）。
- `GITHUB_ACTIONS=true make ci-preflight`：全链路通过——`check_mermaid.cjs`（Mermaid 12.3.0：`7731 diagrams in 7849 Markdown files; 0 errors`）、`generate_link_graph.py`（`7807 nodes, 65091 edges, 21 communities, 614 institutions`，0 孤儿；`exports/link-graph.json` 核验 `roadmap/depth-vla.md` 与全量 `86` 个 Stage 节点 [`80` wiki + `6` roadmap] 均为 `3/3`）、`lint_wiki.py`（`0` 个阻断/常规问题，Sources 覆盖率 `7774/7774 (100%)`；本批 `103` 个改动 `wiki/` 在 `exports/lint-report.md` 检出 `0` 条预警，36 篇 `sources/papers/*.md` 对应 `91` 个目标陈旧失配为 `0`）、`eval_search_quality.py`（`40/40, 100%`）与 `check_export_quality.py`（`15/15`，`7840` 详情页）。
- `make ci-test`（Python 3.12 临时虚拟环境 + Node 22）：`ruff check` 通过、`ruff format --check`（192 files already formatted）、`mypy`（`113` source files 0 问题）、`pip-audit`（No known vulnerabilities found）、`npm run lint:js` 通过、`npm run test:frontend`（`130/130` 通过）、`pytest`（`571 passed, 843 subtests passed`，覆盖率 `63.16%`）。
- 本地静态站无头 Chrome 验证（`127.0.0.1:8766` 与 `127.0.0.1:8767`）：`roadmap.html?id=roadmap-depth-vla` 加载完成（`6 个阶段 · 88 个知识节点 · 63 条阶段内连线`），路线与节点均呈 `3/3`（`below3Labels = []`，`0` 条 `pageerror`/`console.error`），并完成 `.cursor-artifacts/screenshots/roadmap-depth-vla-health-3of3.png` 及 `entity-paper-rtcf` / `entity-paper-neural-introspection-gating` / `entity-paper-tau0-vla` / `wiki-concepts-safe-real-world-rl-fine-tuning` 等 `5` 张本地截图（均呈 `2026-11-14` 与 `3/3` 满格）。
