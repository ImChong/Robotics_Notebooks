# nvidia-isaac.github.io/video_to_data/hydra-0（Hydra-0 项目页）

> 来源归档（ingest）

- **标题：** Hydra-0: Action Flow for Generalist World Modeling and Control
- **类型：** site / project-page
- **官方入口：** <https://nvidia-isaac.github.io/video_to_data/hydra-0/>
- **论文：** <https://arxiv.org/abs/2608.18077>
- **入库日期：** 2026-08-20
- **一句话说明：** NVIDIA Isaac video-to-data 子站：action flow 跨具身 WM、RoboLab 开环 r=0.96、四步蒸馏 16× 加速与逆向 world action model 演示。

## 开源核查（步骤 2.5）

| 资源 | 入库日状态 |
|------|------------|
| 项目页 | 可访问；Overview 视频、定性 rollout、评测表 |
| GitHub / 权重 / 数据 | **未列出** |
| Isaac Lab 部署描述 | 方法说明；**无公开训练仓** |

**结论：** **确认未开源**。

## 页面关键数字

- vs Cosmos 2.5 native action：robot EPE **−90.4%**，object EPE **−60.2%**
- RoboLab 300 episodes：**r=0.96**，mean abs error 5.7 pp
- 训练数据：**2,202 h** 过滤多具身视频
- LightX2V 四步蒸馏：**16.0×** 生成加速（相对多步）

## 对 wiki 的映射

- [`wiki/entities/paper-hydra-0.md`](../../wiki/entities/paper-hydra-0.md)
- [`sources/papers/hydra_0_arxiv_2608_18077.md`](../papers/hydra_0_arxiv_2608_18077.md)
