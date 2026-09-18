# LaWAM（Latent World Action Models）

> 来源归档（ingest）

- **标题：** LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies
- **类型：** paper
- **venue：** CoRL 2026（官方 README 标注）
- **原始链接：** <https://arxiv.org/abs/2606.15768>
- **项目页：** <https://rlinf.github.io/LaWAM/>
- **代码：** <https://github.com/RLinf/LaWAM>
- **机构：** 清华大学；吉林大学；南开大学；北京大学；哈尔滨工业大学；中关村人工智能研究院；跨步智能（Striding.AI）；无问芯穹（Infinigence AI）
- **作者：** Jialei Chen、Kai Wang、Kang Chen、Shuaihang Chen、Feng Gao、Wenhao Tang、Zhiyuan Li、Weilin Liu、Zhuyu Yao、Boxun Li、Yuanbo Xu†、Chao Yu†
- **权重 / 数据：**
  - <https://huggingface.co/jialei02/lawam-libero-sft-lerobot>
  - <https://huggingface.co/datasets/jialei02/libero_merged_no_noops_20hz>
  - <https://huggingface.co/collections/jialei02/lawam-checkpoints>
- **入库日期：** 2026-09-18
- **一句话说明：** 在冻结 DINOv3 潜空间用 **LaWM** 预测单帧 latent visual subgoal 条件 VLA 动作块，LIBERO **98.6%**、RoboTwin **91.22%**、真机 **90.0%** 均值，A100 每 chunk **187 ms**（相对像素 WAM 最高 **24×** 加速）；**LeRobot 原生集成**。

## 核心摘录（MVP）

### 1) 像素 WAM 的冗余与延迟

- **摘录要点：** 现有 WAM 依赖未来图像/视频迭代生成，墙钟延迟高且像素变化对选下一动作块非必需。LaWAM 保留「未来条件控制」但把预测移到预训练视觉编码器潜空间：一个 horizon subgoal 即可条件 Alternate-DiT 动作专家。
- **对 wiki 的映射：**
  - [LaWAM](../../wiki/entities/paper-lawam.md) — 问题设定
  - [World Action Models](../../wiki/concepts/world-action-models.md) — 潜空间 vs 像素 WAM 对照

### 2) 两阶段：LaWM 学习 + 策略蒸馏

- **摘录要点：** Stage 1 在冻结 DINOv3 特征空间训练 latent-action-conditioned **LaWM**（逆动力学编码连续潜动作，前向解码器保留为 world model）；Stage 2 用 **latent-action distillation** 教 VLA 从当前观测+语言预测 transition code，测试时一次非迭代 LaWM 前向产出 subgoal 再生成 chunk。
- **对 wiki 的映射：**
  - [LaWAM](../../wiki/entities/paper-lawam.md) — 方法与流程图
  - [LeRobot](../../wiki/entities/lerobot.md) — 官方 `lawam.mdx` 集成

### 3) 评测与延迟

- **摘录要点：** LIBERO 平均 **98.6%**（187 ms/chunk，2.3B）；RoboTwin 50 任务 **91.22%**（clean 92.52 / randomized 89.48）；真机 Franka + Quanta X1 三任务均值 **90.0%**。相对 Cosmos-Policy、LingBot-VA 等像素 WAM 在 latency–success 帕累托上占优。
- **对 wiki 的映射：**
  - [LaWAM](../../wiki/entities/paper-lawam.md) — 评测读法
  - [GlanceWAM](../../wiki/entities/paper-glancewam.md) — 另一条 WAM 低延迟路线

### 4) 开源状态（截至 2026-09-18）

- **摘录要点：** **已开源** — `RLinf/LaWAM` 含 `starVLA/` 训练、`latent_action_model/` LaWM、`deployment/` 策略服务、LIBERO/RoboTwin 自动评测脚本；HF 发布 LAM、pretrain、LIBERO/RoboTwin SFT 检查点与 LeRobot 格式 LIBERO 数据；**2026-09 已并入 LeRobot 官方文档**。
- **对 wiki 的映射：**
  - [lawam 仓库](../repos/lawam.md)
  - [LaWAM 项目页](../sites/lawam-rlinf-github-io.md)

## 当前提炼状态

- [x] arXiv 摘要、项目页与 README 已对齐
- [x] 仓库 / HF 已交叉核查（步骤 2.5）
- [x] wiki 映射：`wiki/entities/paper-lawam.md`
