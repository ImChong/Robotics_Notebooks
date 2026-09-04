# The Imitator Game 项目页

> 来源归档

- **标题：** The Imitator Game
- **类型：** site / project-page
- **链接：** <https://imitator-game.github.io/>
- **论文：** <https://arxiv.org/abs/2608.22301>
- **代码：** <https://github.com/imitator-game/The-Imitator-Game>（MIT）— 归档见 [`sources/repos/the-imitator-game.md`](../repos/the-imitator-game.md)
- **数据：** <https://huggingface.co/datasets/imitator-game/IG-10K-Dataset> — 归档见 [`sources/datasets/ig-10k.md`](../datasets/ig-10k.md)
- **机构：** 香港大学（HKU）；超忆（TranscEngram）；复旦大学（Fudan）；浙江大学（ZJU）
- **入库日期：** 2026-08-30
- **再核日期：** 2026-09-04
- **一句话说明：** L0–L3 层级、IG-10K、排行榜、Imitator Arena 盲测、文档与数据下载入口。

## 开源核查（步骤 2.5，2026-09-02）

| 组件 | 状态 |
|------|------|
| 项目页 | 有 — 四级定义、实验图、排行榜、Arena、Gallery、Submit、Community |
| 文档 | <https://imitator-game.github.io/docs/guide.html> |
| 数据页 | <https://imitator-game.github.io/data.html> |
| 代码 | **已开源（MIT）** — [imitator-game/The-Imitator-Game](https://github.com/imitator-game/The-Imitator-Game)；站点仓 `imitator-game/imitator-game.github.io` |
| 数据 | **已发布** — HF `imitator-game/IG-10K-Dataset`（约 747 GB，LeRobot-0.5.0）+ `IG-10K-Assets`；ModelScope `Zhouxunzhe/IG-10K-Dataset` / `IG-10K-Assets` |
| 权重 | HF org 截至再核日 **0 个公开模型**；基线需按仓内 recipe 自训 |
| 真机评测 | Arena 开放盲测；真机部署走社区申请，仓内无上传接口 |

2026-08-30 浅入库时页上未见独立训练仓，判定为「部分开源」。再核时官方仓、数据集与资产均已公开，改判 **已开源**。

## 页上可核事实（非 PDF）

- 卖点口径：四级意图模仿、20,000+ 配对、可扩展仿真、Arena 人类评测。
- 固定协议：5 seen + 5 unseen × L0–L3；完整 50 任务表在 leaderboard 作参考。
- 实验轴：接口族（视频 vs VLA）、编码器、语料规模、层级剖面、自动指标 vs Arena。
- 社区入口：Discord / 微信；欢迎提交任务、模型与本体。

## 对 wiki 的映射

- [Imitator Game 实体页](../../wiki/entities/paper-imitator-game.md)
- [论文摘录](../papers/imitator_game_arxiv_2608_22301.md)
