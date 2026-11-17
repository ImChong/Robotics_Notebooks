# NVIDIA 公司路线健康度审计（2026-11-17）

范围：`company.html?id=nvidia`；公司时间线 31 → 36 个事件，**35 个唯一 wiki 详情节点**（Isaac Lab 首发与 3.0 GA 复用一个详情）。不修改评分函数、draft/stub 封顶、PR 门禁或生成统计。

## 处理结果

- 更新前：**26/35** 个唯一节点为 3/3，其中 9 个旧节点原文未通读或阅读骨架不齐。
- 本次补核 9 个节点及其原文归档；每个 wiki 页新增 `original:`，对应 `sources/` 的一手材料记录一项 `- **通读：** 已完成`；缺失小节及必要流程图已补齐。
- 按 `scripts/generate_link_graph.py` 的三项判据（元数据、阅读骨架、原文通读）核对：**35/35 个唯一节点具备 3/3 条件**，待 GitHub Actions 的健康度报表验证。
- 原有 5 个本 PR 更新的节点此前已为 3/3，本轮无须改动其结论；新增核读项另见下表。
- **核读与运行界限：** 对照论文一手 HTML/官方 README、文档、项目页及可见运行入口进行内容核验；**未运行 GPU 训练、推理、真机实验**。当项目页没有可核源码运行入口（如 Cosmos Policy、Vesta）时明确写明，不捏造时序图。

## 本轮九项核读结果

| 页面 | 健康度 | 缺项 |
|---|---|---|
| wiki/entities/mimicgen.md | 3/3 | — |
| wiki/entities/paper-notebook-dreamgen-unlocking-generalization-in-robot-learn.md | 3/3 | — |
| wiki/entities/paper-sa-2511-00062-world-simulation-with-video-foundation-models-fo.md | 3/3 | — |
| wiki/entities/paper-doorman-opening-sim2real-door.md | 3/3 | — |
| wiki/entities/paper-shenlan-wm-11-cosmos-policy.md | 3/3 | — |
| wiki/entities/paper-hrl-stack-35-dreamdojo.md | 3/3 | — |
| wiki/entities/kimodo.md | 3/3 | — |
| wiki/entities/paper-motionbricks.md | 3/3 | — |
| wiki/entities/paper-vesta-generalist-embodied-reasoning.md | 3/3 | — |

## 原有五项核读结果

| 页面 | 健康度 | 缺项 |
|---|---|---|
| wiki/entities/isaac-gym.md | 3/3 | — |
| wiki/entities/curobo.md | 3/3 | — |
| wiki/entities/cosmos-curator.md | 3/3 | — |
| wiki/entities/cosmos-cookbook.md | 3/3 | — |
| wiki/entities/paper-bfm-14-hover.md | 3/3 | — |

## 重点复核结论

- **MimicGen：** 原论文 18 任务 / 50K+ 轨迹，与开源 README 12 任务 / 48K+ 数据分开标注。原文归档：[`sources/papers/mimicgen_arxiv_2310_17596.md`](../../sources/papers/mimicgen_arxiv_2310_17596.md)。
- **MotionBricks：** [独立官方仓 NVlabs/motionbricks](https://github.com/NVlabs/motionbricks) 已发布 SOMA/G1 推理运行时、checkpoint 和示例，替换早期“只有预览代码”的历史判断；完整训练资产与运行时开放分别说明。
- **DreamDojo：** 区分人视频数据与混合总量，不将蒸馏后视频 FPS 当作机器人控制频率。
- **Cosmos Policy / Vesta：** 文档给出算法与官方论文推理流程，但未验证完整发布的可运行官方源码入口，因此不把论文流程标为已运行源码。
- **原有最新路线核查：** [官方资料核验](../../sources/blogs/nvidia_robotics_latest_2026-11-17.md)；Isaac Lab 3.0、cuRoboV2-Fast、GR00T 2 与 DexBench 事件仍按各自发布时间与开放边界区分。

## 后续验证

在 PR #3847 的最新提交上运行 `python3 scripts/report_health.py --changed`、`make ci-preflight`、`make test` 与 Mermaid 全量检查，检查 PR 描述所列全部变更页的健康度是否与 CI 一致。提交未改变 `docs/company-roadmaps.json` 的节点身份映射。
