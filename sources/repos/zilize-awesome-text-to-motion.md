# Zilize / awesome-text-to-motion

- **URL（仓库）**: https://github.com/Zilize/awesome-text-to-motion（默认分支 `master`）
- **URL（项目页）**: https://zilize.github.io/awesome-text-to-motion/ — Plotly 交互图与统计视图，浏览体验优于纯 README
- **Maintainer**: Zilize（GitHub）
- **定位**: 文本驱动 **单人** 人体运动生成方向的 **综述 / 数据集 / 模型** 精选清单；README 明确 **不包含** 人–物 / 场景交互类设定（与 broader HOI motion 资源刻意切分）

## 仓库结构（维护者视角）

- `data/arxiv.csv` — 带 arXiv ID 的论文元数据行（survey / model / dataset 标记、`backbone_tags`、`approach_tags` 等）
- `data/without-arxiv.json` — 无 arXiv ID 的条目 JSON
- GitHub Actions 静态站 — 将上述数据渲染为图表与筛选界面

## README 主干目录（知识地图）

### Surveys（示例）

- *Motion Generation: A Survey of Generative Approaches and Benchmarks* — arXiv:2507.05419
- *Multimodal Generative AI with Autoregressive LLMs for Human Motion Understanding and Generation* — arXiv:2506.03191
- *Text-driven Motion Generation: Overview, Challenges and Directions* — arXiv:2505.09379
- *Human Motion Generation: A Survey* — TPAMI 2023 / arXiv:2307.10894

### Datasets（与机器人知识库交叉较多的锚点）

- **HumanML3D** — CVPR 2022 文本–运动对数据，领域内常用基准之一
- **KIT-ML** — KIT Motion-Language Dataset（Big Data 2016）
- **Motion-X / Motion-X++** — 大规模全身表现型人体运动
- **HumanML3D++ / HumanML3D-Extend** — 更长或更开放文本条件下的扩展任务
- 其余：FineMotion、SnapMoGen、MotionMillion、MotionFix、MotionLib 等（详见原 README **Datasets** 节）

### Models（与 NVIDIA / 扩散范式相关的锚点）

- **GENMO** — arXiv:2505.01425（本库已单独 ingest）
- **MotionGPT3**、**Motion-R1**、**MoMADiff**、**ReMoMask** 等 — 覆盖自回归、扩散、检索增强、偏好对齐等路线（详见原 README **Models** 节）

## 对 wiki 的映射

- 作为 **人体文本–运动（T2M）** 领域的 **外部索引入口**，沉淀为实体页 [awesome-text-to-motion-zilize](../../wiki/entities/awesome-text-to-motion-zilize.md)
- 与机器人侧 **扩散运动生成**、**SMPL 系重定向** 页面交叉引用，便于从「动画/人体生成」跳转到「控制与重定向」
