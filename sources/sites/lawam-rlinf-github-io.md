# LaWAM 项目页（rlinf.github.io/LaWAM）

> 来源归档

- **标题：** LaWAM: Latent World Action Models
- **类型：** site
- **链接：** <https://rlinf.github.io/LaWAM/>
- **论文：** <https://arxiv.org/abs/2606.15768>
- **代码：** <https://github.com/RLinf/LaWAM>
- **入库日期：** 2026-09-18
- **一句话说明：** 官方项目页：潜空间 subgoal vs 像素 WAM 对比、两阶段管线图、LIBERO/RoboTwin/真机数字与演示视频。

## 步骤 2.5 开源核查（2026-09-18）

- **已开源：** 页内链到 GitHub `RLinf/LaWAM` 与 Hugging Face 模型集合 / 数据集（LIBERO merged、RoboTwin merged、SFT 检查点）。
- **LeRobot：** README 标注 2026-09 官方集成 [`lerobot` lawam 文档](https://github.com/huggingface/lerobot/blob/main/docs/source/lawam.mdx)。
- **沉淀到 wiki：** [`wiki/entities/paper-lawam.md`](../../wiki/entities/paper-lawam.md)

## 页内关键数字（复核用）

| 基准 | 成功率 / 延迟 |
|------|----------------|
| LIBERO（10 denoise steps, A100） | **98.6%**，**187 ms**/chunk |
| RoboTwin（50 任务 × 100 trials） | **91.22%** overall |
| 真机（3 任务 × 30 trials） | **90.0%** average |

对照表列 pi0.5、Cosmos-Policy、LingBot-VA 等同窗 latency–SR 点。
