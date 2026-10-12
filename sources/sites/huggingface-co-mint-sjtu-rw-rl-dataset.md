# RW-RL-Dataset（Hugging Face：MINT-SJTU/RW-RL-Dataset）

> 来源归档

- **标题：** RW-RL Dataset
- **类型：** site（数据集卡）。周报把该链接当作“项目页”。
- **URL：** <https://huggingface.co/datasets/MINT-SJTU/RW-RL-Dataset>
- **论文：** <https://arxiv.org/abs/2610.13345>（归档见 [evo_rl_arxiv_2610_13345.md](../papers/evo_rl_arxiv_2610_13345.md)）
- **代码：** <https://github.com/MINT-SJTU/Evo-RL>（归档见 [evo-rl.md](../repos/evo-rl.md)）
- **入库日期 / 核查日期：** 2026-10-12
- **一句话说明：** Evo-RL 配套真机强化学习数据集：遥操作、人类干预、自主 rollout，含奖励与 done 信号。

## 页面核查结论

- 卡片称：浏览器显示 111,281 行，总计 1.88 TB，三批发布共 23,303 个 episode；规划目标 1000+ 真机小时。
- 本体：R1Lite 13,642 episode / 5 任务 / 4 相机 / 15 FPS；G2A 5,211 / 5 / 8 / 30 FPS；SO101 双臂 4,450 / 6 任务（约 224.7 小时）/ 3 相机 / 30 FPS / 12 维动作与状态。卡片提到 PiperX，但发布表没有 PiperX 目录（与论文 PiperX 实验的对应关系未核实）。
- 格式：LeRobot 风格，每 episode 一个 parquet + 每相机一个 MP4，含 `meta/info.json`、`tasks.parquet`；约 11 万个视频文件。
- 许可：卡片未写；论文引用：卡片称之后补充正式引用；配套人类干预数据集 BodenAI/RW-RL-HIL-Dataset。
- 判定：**数据已开放**；论文中的 PiperX 实验数据是否全部包含未核实。

## 对 wiki 的映射

- [Evo-RL 详情节点](../../wiki/entities/paper-evo-rl.md)
