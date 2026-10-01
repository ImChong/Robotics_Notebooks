# ChunkTrust 官方项目页（hf618.github.io）

> 来源归档

- **标题：** ChunkTrust — Adapting Execution Horizons for Robot Policies with Action-Expert Evidence
- **类型：** site / project-page
- **URL：** <https://hf618.github.io/ChunkTrust.github.io/>
- **关联论文：** <https://arxiv.org/abs/2609.39754>
- **代码：** <https://github.com/hf618/ChunkTrust>（页内链 GitHub；MIT）
- **Hugging Face：** <https://huggingface.co/Niugan/ChunkTrust>（README badge；QHA checkpoint）
- **机构：** 清华大学；北京智源人工智能研究院（BAAI）；中国人民大学等（见页内作者列表）
- **入库日期：** 2026-10-01

## 步骤 2.5 开源核查（2026-10-01）

| 项 | 结论 |
|----|------|
| 项目页 / README GitHub | **hf618/ChunkTrust**，含安装、`examples/quickstart.py`、`docs/integration.md` |
| Hugging Face | **Niugan/ChunkTrust** — QHA head 与 `scripts/download_asset.py` 下载脚本 |
| Base VLA 权重 | **非本仓分发**；RoboTwin / RoboCasa / OpenPI 等按 backend 文档自备 |
| **判定** | **已开源（AHS 库 + QHA 权重 + 评测配置）**；策略 backbone 走各 benchmark 原链路 |

## 页面结构归纳

1. **Teaser：** 阶段依赖的最优 replan horizon；RoboTwin2.0 / RoboCasa / 真机相对提升 headline。
2. **Evidence：** 谱稳定性 + 运动连续性双信号；失败 episode 统计动画。
3. **Method：** AHS 在线评分 + QHA 先验摊销；phase-aware memory。
4. **Results：** 仿真表（多 policy family）；真机 rollout 视频与阶段–horizon 可视化。
5. **Integration：** 强调 evaluation loop **drop-in wrapper**，不改 policy 权重。
