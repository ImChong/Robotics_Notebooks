# Fast-VLA 论文来源归档

> 来源归档（ingest）

## 元数据

- **题目：** Fast-VLA: You Only Think Once for Robot Manipulation
- **作者：** Kehong Gong、Guanli Hou、Xiaoxu Zheng、Qi Wang、Mingxi Xu、Anbang Zhai、Xiaoyu He、Mingyuan Zhang、Michael Bi Mi
- **机构：** 华为、华为中央研究院
- **预印本：** <https://arxiv.org/abs/2610.21389>
- **项目页：** <https://garfield-kh.github.io/FastVLA/>
- **代码：** <https://github.com/Garfield-kh/FastVLA>
- **模型：** <https://huggingface.co/kehong/FastVLA-weights>
- **版本：** v1，2026-10-15
- **入库日期：** 2026-10-19
- **一句话说明：** 每个 episode 只调用一次冻结 VLM 建立任务 KV 记忆，后续由轻量 actor 闭环生成动作块。

## 核心论文摘录

### 方法

Fast-VLA 将任务理解与快速控制解耦。冻结的 4B VLM 读取指令和初始视角，生成跨层 key/value 任务记忆；每个动作块由在线 actor 融合当前头部/腕部图像、本体状态和时间/手臂索引查询，再从固定记忆读取任务上下文，不重复运行基础 VLM。

### 评测与结果

- 项目页报告 RoboTwin 2.0 Clean/Randomized 成功率 92.46%/92.40%，平均摊销推理耗时 1.437 ms/步，在线常驻内存 1.22 GiB。
- LIBERO / LIBERO-Plus 成功率为 95.95%/79.11%，摊销推理耗时 3.975 ms/步。
- 同硬件对比中，RoboTwin 推理时间较所选持续条件化基线低 10.9–17.1 倍；此效率依赖 episode 初始视图能建立足够可靠的固定任务记忆。

## 对 wiki 的映射

- [Fast-VLA 实体页](../../wiki/entities/paper-fast-vla-once-per-episode.md)
- [官方项目页归档](../sites/fast-vla.md)
- [官方仓库归档](../repos/fast-vla.md)

## 一手来源

- [arXiv 摘要](https://arxiv.org/abs/2610.21389)
- [项目页](https://garfield-kh.github.io/FastVLA/)
- [代码与模型](https://github.com/Garfield-kh/FastVLA)
