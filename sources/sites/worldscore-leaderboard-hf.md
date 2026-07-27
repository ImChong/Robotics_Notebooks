# WorldScore Leaderboard（Hugging Face Space）

> 来源归档（ingest）

- **标题：** WorldScore Leaderboard
- **类型：** site / leaderboard（Hugging Face Space）
- **官方入口：** <https://huggingface.co/spaces/Howieeeee/WorldScore_Leaderboard>
- **SDK：** `static`（`index.html` + `script.js` + `style.css` + `leaderboard.csv`）
- **关联论文 / 项目页 / 仓：**
  - <https://arxiv.org/abs/2504.00983>
  - <https://haoyi-duan.github.io/WorldScore/>
  - <https://github.com/haoyi-duan/WorldScore>
- **入库日期：** 2026-07-27
- **一句话说明：** WorldScore 官方可更新排行榜：按 Model Type（Video / 3D / 4D）展示 WorldScore-Static / Dynamic 与十项子指标；社区可自采样自评测后提交 `worldscore.json` 上榜。

## 数据卡要点（2026-07-27 核查）

- Space 元数据：`Howieeeee/WorldScore_Leaderboard`，emoji 📊，`sdk: static`，`app_file: index.html`。
- 主数据文件：`leaderboard.csv`（列含 Model Type、Model Name、Ability、Sampled by、Evaluated by、Accessibility、Date、WorldScore-Static/Dynamic、Camera/Object Control、Content Alignment、3D/Photometric/Style Consistency、Subjective Quality、Motion Accuracy/Magnitude/Smoothness）。
- 截至入库日 CSV **34** 行模型；Static 前列示例：UniWorld-View **85.53**、WorldScape-0.2(MoE) **85.13**、World Dreamer **84.52**；论文首发批次多为 Date `2025.03.30`。
- 提交路径（仓库 README）：推荐 **Your team 采样 + Your team 评测**，跑 `worldscore-analysis -cs` 校验完整后，把 `worldscore_output/worldscore.json` 发至 **haoyiduan@princeton.edu**；也可交视频由官方代评（进度依赖资源）。

## 读榜注意

- **Static 高 ≠ Dynamic 高**：纯 3D 场景生成常因动力学维记 0 而 Dynamic 显著偏低（如 WonderWorld Static 72.69 / Dynamic 50.88）。
- **活榜 ≠ 论文 Table 2**：项目页与论文表是快照；选型与复现请以本 Space CSV 最新行为准，并回查 Accessibility / Sampled by / Evaluated by。
- **与 EWMBench 不同轴**：本榜评开放域 **多场景世界生成**（相机布局 + 质量 + 动态），不是机器人操纵场景守恒 / 末端轨迹基准。

## 对 wiki 的映射

- [WorldScore（论文实体）](../../wiki/entities/paper-worldscore.md) — 「实验与评测 / Leaderboard」节
- [haoyi-duan-worldscore-github-io.md](./haoyi-duan-worldscore-github-io.md)
- [worldscore.md](../repos/worldscore.md)
- [worldscore_arxiv_2504_00983.md](../papers/worldscore_arxiv_2504_00983.md)
