# 目前最接近具身 ICL 如何实现的必读论文解读

> 来源归档（blog / 微信公众号）

- **标题：** 目前最接近具身 ICL 如何实现的必读论文解读
- **类型：** blog
- **作者：** 每日智能（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/vIUalf3vZI3AV-HWSVruew
- **发表日期：** 2026-08-31
- **入库日期：** 2026-08-31
- **抓取方式：** `wechat-article-for-ai`（Camoufox）；`--no-images`
- **原始抓取落盘：** [`sources/raw/wechat_meiri_zhineng_embodied_icl_four_papers_2026-08-31/`](../raw/wechat_meiri_zhineng_embodied_icl_four_papers_2026-08-31/)
- **一句话说明：** 纵向剖析 + 横向对比 WAM-TTT、RoboTTT、StellaVLA、Zero-WAM 四条「演示当提示」路线；强调跨篇数字不可直接比较，并给出漂移轴地图与七条「未走之路」。

## 核心摘录（归纳，非全文）

### 坐标系：演示改什么 × 何时适应

| | **改权重（快权重）** | **改输入（纯上下文）** |
|--|---------------------|------------------------|
| **部署前一次** | WAM-TTT（感知侧技能记忆） | StellaVLA、Zero-WAM |
| **rollout 每步** | RoboTTT（动作侧工作记忆） | — |

- **WAM-TTT vs RoboTTT：** 前者快权重是「技能包」、部署前写好即冻结；后者是「工作记忆」、每步更新。
- **产业对照：** GEN-1.5 / S1 宣称涌现式 ICL（闭源）；四篇论文均显示 **显式机制**（meta-training、sequence forcing、结构化管线、IFP）不可或缺。

### 各篇关键洞见（策展）

**WAM-TTT（2607.06988）**

- KVM 损失 ≈ 无 softmax 线性注意力；概念上缝合「演示当上下文」与「演示当记忆」。
- 去掉 meta-training 几乎归零；通用 LoRA 换不掉 TTT 快权重；加伪动作净负面。
- 配对人类数据可 1:1 替代机器人数据（等预算 200 ep）；但 (10,190) 仅 51.4 → 人类数据是补充非替代。
- **折扣：** progress 为部分给分；零梯度指零**主干**梯度；无代码/项目页。

**RoboTTT（2607.15275）**

- 主命题是 **上下文长度 scaling 轴**；8K 为**预训练**上下文，主结果部署策略在 1K 后训练。
- one-shot 人视频为**机器人固定相机**拍摄、机器人静止；「未见」= 80 种元件构型里的未见组合。
- DAgger Distillation：失败作 context、纠正作 target，+33% vs 标准 DAgger +9%。
- **折扣：** 未与全注意力长上下文或专用 ICL 方法对比；+1 历史帧反而有害（39.5 vs 57）。

**StellaVLA（2608.11671）**

- 四篇中**唯一完全零梯度**；离线 Qwen3-VL 结构化管线（子目标 + 2D/3D 运动 verbalization）。
- **三向干预：** 正确 98.8 / 无 62.4 / 错 44.9 — 错比无更差，证明策略主动用上下文。
- Text-only 98.8/84.4 ≈ Image+Text；Image-only OOD 差 6.3 pt。
- λ 消融：LIBERO-Plus OOD 在 **λ=0 最高（86.9）**；λ=0.3 为 85.1 → 语言监督可能伤 OOD。
- **折扣：** 仿真检索≈精确任务匹配；Long Horizon L1/L2 近零；算力未披露。

**Zero-WAM（2608.26103）**

- 唯一正面攻 **未见任务**（RoboTwin 7 任务留出）；HumanGen 74.2K 对 / 8.6K 任务。
- **IFP** 防 teacher-forcing 捷径；IFP **不直接以人类视频为条件**（推理时移除）。
- 跨消融推算：无 IFP 时加人视频（28.55）< 不加（39.44）；完整 46.95。
- **折扣：** 训练人视频 100% 合成；质检无通过率数字；LingBot-VA 为同组前作。

### 三条跨篇经验

1. **人类信息只改感知/生成侧，不直接碰动作侧**（WAM-TTT 视频专家、Zero-WAM 视频分支）。
2. **训练期辅助推理、推理期剥离**（StellaVLA spatial-language expert、Zero-WAM IFP）；辅助目标须监督主分支。
3. **上下文非免费午餐** — 历史帧、Image-only 示范、无 IFP 的人视频均可走捷径。

### 七条「未走之路」（文内优先级）

1. 三方对照：结构化文本 vs 合成视频 vs 快权重记忆（**边际信息量最高**）
2. 常量前缀 + 递归状态叠加（Zero-WAM × RoboTTT）
3. KVM 损失接到 RoboTTT
4. 强全注意力长上下文 ICL 基线
5. 合成训练 / 真实测试人视频 gap
6. 自动标注质检数字
7. human-robot gap 三种解法同台比较

## 对 wiki 的映射

- **对比页（新建）：** [wam-ttt-robottt-stellavla-zero-wam-embodied-icl](../../wiki/comparisons/wam-ttt-robottt-stellavla-zero-wam-embodied-icl.md)
- **实体页交叉更新：** [WAM-TTT](../../wiki/entities/paper-wam-ttt-human-video-test-time-steering.md)、[RoboTTT](../../wiki/entities/paper-robottt-test-time-training-vla-context.md)、[StellaVLA](../../wiki/entities/paper-stellavla-structured-icl-vla.md)、[Zero-WAM](../../wiki/entities/paper-zero-wam.md)
- **概念页：** [robot-in-context-learning](../../wiki/concepts/robot-in-context-learning.md)
