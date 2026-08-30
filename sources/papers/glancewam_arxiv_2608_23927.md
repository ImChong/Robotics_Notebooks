# GlanceWAM（稀疏测试时想象）

> 来源归档（ingest）

- **标题：** GlanceWAM: Sparse Test-Time Imagination for World-Action Models
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.23927>
- **机构：** 弗吉尼亚理工学院（Virginia Tech）；德雷塞尔大学（Drexel University）；美国东北大学（Northeastern University）；普渡大学（Purdue）
- **作者：** Linhan Wang、Zijian An、Mingyuan Zhang、Chen Dai、Yi Xu、Can Cui、Zichong Yang、Yinlin Chen、Lifeng Zhou、Chang-Tien Lu
- **代码：** <https://github.com/linhanwang/GlanceWAM>
- **权重 / 数据：** <https://huggingface.co/datasets/LinhanWang/GlanceWAM>
- **入库日期：** 2026-08-30
- **一句话说明：** 单视频 DiT 内把视觉想象移出控制关键路径：异步 proposer 后台生成约 3 s 后的单帧前瞻，动作头在潜空间以 48 ms 解码动作块。

## 核心摘录（MVP）

### 1) 同步视频生成 vs 取消想象的两难

- **摘录要点：** 控制频率同步生成视频延迟不可接受；完全取消测试时视觉想象又掉成功率。GlanceWAM 证明：想象只要异步、离关键路径、并在潜空间消费，就能同时保住实时性与成功率。
- **对 wiki 的映射：**
  - [GlanceWAM](../../wiki/entities/paper-glancewam.md) — 问题设定。
  - [World Action Models](../../wiki/concepts/world-action-models.md) — Joint WAM 延迟对照。

### 2) 非干扰注意力 + 抗陈旧时域训练

- **摘录要点：** 非干扰三分类注意力掩码隔离视频表示，避免泄漏进动作读点；staleness-robust horizon 训练让前瞻帧在刷新间隔内老化仍可用。骨干 SkyReels-V2 DF；动作头 GR00T 风格流匹配，0.8 s chunk。
- **对 wiki 的映射：**
  - [GlanceWAM](../../wiki/entities/paper-glancewam.md) — 方法。
  - [Action Chunking](../../wiki/methods/action-chunking.md) — 异步 chunk 部署。

### 3) 评测数字

- **摘录要点：** 仅用示范训练。RoboCasa kitchen 24 项 **72.2%**（同步 Cosmos Policy 67.1%，无想象共训 64.4%）；LIBERO **99.0% / 0.989**。A100 每 chunk **48 ms**，比同步基线快 **24×**。
- **对 wiki 的映射：**
  - [GlanceWAM](../../wiki/entities/paper-glancewam.md) — 评测读法。
  - [VLA](../../wiki/methods/vla.md) — 世界模型条件动作头。

### 4) 开源状态（截至 2026-08-30）

- **摘录要点：** **已开源** MIT。`glancewam/training/train.py`、LIBERO / RoboCasa sweep 评测、HF 21 GB 数据+检查点齐备。框架自 [StarVLA](https://github.com/JinhuiYE/starVLA) 抽出。
- **对 wiki 的映射：**
  - [glancewam 仓库](../repos/glancewam.md)

## 当前提炼状态

- [x] arXiv 摘要、方法与评测节已对齐摘录
- [x] 仓库 / HF 已交叉核查
- [x] wiki 映射：`wiki/entities/paper-glancewam.md` 新建
