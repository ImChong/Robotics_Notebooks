# Argus 官方技术文章

- **标题：** Argus: An Open-Source Annotator for Robotics Data
- **类型：** site / 官方技术文章
- **作者 / 机构：** Eric Li / Pantheon
- **来源日期：** 2026-10-01
- **入库日期：** 2026-10-02
- **链接：** https://pantheon.inc/research/argus
- **代码：** https://github.com/Pantheon-Industries-Inc/argus（已开源）
- **数据面板：** https://pantheon.inc/data-board
- **在线工具：** https://data.pantheon.inc/review
- **前篇：** https://pantheon.inc/research/we-looked-at-the-data
- **中文转述：** https://mp.weixin.qq.com/s/MttdEbIP__Y_ApGp5_4ZuQ
- **交叉归档：** [源码与运行入口](../repos/pantheon_argus.md)
- **沉淀到 wiki：** [Argus](../../wiki/entities/pantheon-argus.md)

## 值得保留的观察

1. 多相机视频、运动信号和文字指令必须交叉核验：指令本身可能错误，腕部相机运动也可能被误认成物体运动。
2. 审计样本为九个数据集的 3,546 个片段、66.5 小时；其中 27% 至少有一项中/高严重度问题。此为被审计片段的统计，不能当作所有机器人数据的普遍缺陷率。
3. 目标首次完成后可能又被破坏；goal frame 有助于重建有效成功区间，而不能只看录像末帧。
4. 行为克隆应裁剪或过滤演示失误；世界模型可保留真实失败与恢复，但需明确标注。错误时间基准、相机互换等录制故障应先修复。
5. 与参考模型结果一致、标注事件更多，均不等于人工核验准确率更高。

## 开源核查

官方文章明确指向 Pantheon-Industries-Inc/argus；仓库 README 提供 prepare/checks/label/review/board 可运行入口，并声明代码 Apache-2.0、公开标注 CC BY 4.0。面板可逐片段下载 JSON、按筛选导出 JSONL。原始数据及可选手部关键点保留各自授权边界。
