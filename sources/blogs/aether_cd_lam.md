# CD-LAM: Causal Debiasing Gives World Models Stronger Action Control with 10x Less Post-training（Aether AI 博客）

> 来源归档（blog / Aether AI 官方 Field notes #07）

- **标题：** CD-LAM: Causal Debiasing Gives World Models Stronger Action Control with 10x Less Post-training
- **类型：** blog（论文解读文 + 讲解视频，配套 arXiv 论文）
- **作者 / 组织：** 页面署名「Team Wei, Zhou, Huang」/ Aether AI 博客（aetherlabs.ai）
- **原始链接：** <https://aetherlabs.ai/articles/cd-lam-causal-debiasing-for-embodied-world-models.html>
- **博客索引：** <https://aetherlabs.ai/blog.html>（编号 07 · Causal Debiasing；标签 Causal AI · Embodied AI · World Models）
- **发表日期：** 2026-07-27
- **入库日期：** 2026-10-10
- **抓取方式：** `curl` 抓取静态 HTML 后抽取正文与表格
- **一句话说明：** 基于重建训练的潜动作模型（LAM）会把背景运动、相机漂移等与动作无关的因素编进潜动作（作者称 **视觉混杂**），导致下游动作条件世界模型「视频逼真但不听动作」。CD-LAM 在 LAM 阶段加三项去偏目标，不改世界模型骨干和动作接口；相对 DreamDojo，动作跟随误差降 30% 以上，机器人动作后训练约 3k 步追平基线 50k 步。

## 开源 / 项目页核查（步骤 2.5，截至 2026-10-10）

| 项 | 结论 |
|----|------|
| 论文 | arXiv:2607.09185（见 [论文归档](../papers/cd_lam_arxiv_2607_09185.md)） |
| 项目页 | <https://yufanwei.github.io/CD-LAM-project-page/>（可访问） |
| 代码 | **已开源**：博客链 <https://github.com/yufanwei/CD-LAM>；arXiv v2 链 <https://github.com/AetherLabsAI/CD-LAM>，两者 `git ls-remote` HEAD 同为 `5540960`（**推测** 为同一仓库的迁移 / 镜像）；README 标 Apache-2.0 |
| 权重 | HF `yufanwei/CD-LAM`（创建 2026-07-09）与 `AetherLabs-AI/CD-LAM` 均公开；含 `models/lam`、`models/pretrain`、`models/posttrain`（+ `bridge.pt`、`action_contract.json`），**仅 2B**；README 明写 14B 权重与 14B 运行适配器 **不包含** |
| 数据 | 依赖公开数据集 AgiBotWorld Alpha 与 EgoDex（README 提供下载与预处理脚本） |

## 核心摘录（归纳，非全文）

### 问题：逼真 ≠ 因果

- 具身世界模型 = 机器人的想象力：给当前视图 + 计划动作，预测下一帧；用于规划、策略评估、数据增广。需要 **realism** 与 **causality** 两样。
- 现代流水线：LAM 给人类视频打潜动作标签 → 世界模型预训练 → 机器人动作后训练。每一阶段都继承 LAM 的动作空间。
- 重建目标从不区分「这一帧变化是机器人动作、相机运动还是光照引起」，于是把背景连续性、物体外观、相机漂移折进潜动作 → **视觉混杂（visual confounding）**。作者主张「更高的重建精度也去不掉它」。

### 两个干预测试

- **零动作** \(do(u=0)\)：固定首帧、后续动作全置零，理想输出静止。14B 示例：DreamDojo 残余 FDCE **44.2 px**，CD-LAM **3.3 px**（单样本）。
- **目标动作** \(do(u=u_{tar})\)：固定首帧、换入另一条轨迹的动作序列，理想输出跟随新动作。基线几乎不变。

### LAM 编码器诊断（仅编码器，不涉及生成；越低越好）

| 诊断 | DreamDojo LAM | CD-LAM |
|------|---------------|--------|
| 两帧相同时的响应（相对范数中位数） | 0.527 | 0.043 |
| 绝对潜范数中位数 | 3.119 | 0.226 |
| 水平平移鲁棒性（均值 / 中位数） | 0.555 / 0.536 | 0.156 / 0.096 |
| 垂直平移鲁棒性（均值 / 中位数） | 0.545 / 0.529 | 0.110 / 0.064 |
| 场景捷径泄漏（scene-shortcut leakage） | 0.151 | 0.014 |

### 新指标 FDCE

- **FDCE（Foreground Displacement Chamfer Error）**：SAM3 分割前景（机械臂 + 被操作物体）→ CoWTracker 跟踪前景点 → 生成与参考位移轨迹的 Chamfer 距离。
- PSNR 对动作跟随误差的解释力很弱：散点图 \(R^2=0.14\)。PSNR / SSIM / LPIPS 仍作为视觉保真度指标报告。

### 方法：三阶段 + 三项去偏目标

- 阶段 1 LAM 去偏微调 → 阶段 2 动作条件世界模型（ACWM）在去偏潜动作上微调 → 阶段 3 机器人动作后训练（轻量适配器把真实控制命令映射进同一潜动作空间）。骨干、潜动作维度、输入格式都不变。
- \(\mathcal L_{CD}=\mathcal L_{emb}+\lambda_{ctr}(k)\mathcal L_{ctr}+\lambda_{cal}\mathcal L_{cal}\)：
  - **具身中心重建** \(\mathcal L_{emb}\)：SAM3 前景掩码 \(M_t\) 加权重建，\(\alpha_{fg}>\alpha_{bg}\)。
  - **动作中心对比** \(\mathcal L_{ctr}\)：从视频文本标注抽动词归并成动作原语（pick-and-place、pour、open…），SigLIP 式成对 softplus 损失；\(\lambda_{ctr}(k)\) 随训练步变化。
  - **潜空间校准** \(\mathcal L_{cal}=\mathcal L_{KL\text{-}fb}+\mathcal L_{zero}\)：两帧相同时把潜动作范数推向 0（以普通转移潜变量运行 RMS 范数归一、stop-grad），free-bits KL 防塌缩。

### 结果（自报）

**仅潜动作条件（ACWM 去偏微调后，EgoDex 留出人类视频）**

| 模型 | FDCE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 目标动作 FDCE ↓ |
|------|--------|--------|--------|---------|-----------------|
| DreamDojo-2B | 34.00 | 20.88 | 0.780 | 0.413 | 42.74 |
| CD-LAM-2B | 19.63（−42%） | 24.29 | 0.827 | 0.308 | 33.81（−21%） |
| DreamDojo-14B | 40.29 | 21.04 | 0.792 | 0.398 | 50.27 |
| CD-LAM-14B | 29.87（−26%） | 23.18 | 0.814 | 0.342 | 33.22（−34%） |

**机器人动作后训练后（留出真机数据）**

| 模型 | FDCE 均值 ↓ | FDCE 中位 ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 零动作 FDCE ↓ | 目标动作 FDCE ↓ |
|------|-------------|-------------|--------|--------|---------|---------------|-----------------|
| DreamDojo-2B | 12.63 | 8.15 | 19.85 | 0.798 | 0.271 | 10.71 | 24.36 |
| CD-LAM-2B | 8.24（−34.8%） | 6.75 | 20.60 | 0.806 | 0.269 | 5.03（−53%） | 22.55（−7%） |
| DreamDojo-14B | 11.11 | 8.98 | 20.01 | 0.808 | 0.263 | 9.36 | 24.82 |
| CD-LAM-14B | 7.73（−30.4%） | 5.99 | 21.01 | 0.818 | 0.247 | 2.18（−76.7%） | 21.11（−15%） |

- 基线从 2B 放大到 14B 并不消除漂移，潜动作生成上误差反而随规模变大；作者据此认为收益来自去偏而非规模。
- **训练效率**：CD-LAM 在 3,000–4,000 步追平 DreamDojo 50,000 步的参考点（博客口径「>10×」）。
- **数据效率**（2B，阶段 1 固定 1,000 步）：

| 去偏数据量 | PSNR ↑ | FDCE 均值 ↓ | FDCE 中位 ↓ |
|------------|--------|-------------|-------------|
| DreamDojo 基线 | 19.85 | 12.63 | 8.15 |
| 1 h | 20.54 | 8.91 | 6.88 |
| 10 h | 20.61 | 8.87 | 6.23 |
| 100 h | 20.60 | 8.24 | 6.75 |
| 1000 h | 20.64 | 7.97 | 6.12 |

- 博客口径：1 小时视频拿到 1,000 小时约 80% 的收益。

## 可信度边界

- 博客（2026-07-27）对应 arXiv v1 口径；arXiv v2（2026-09-26）摘要改为「>12× 更少适配更新」、机器人动作下 FDCE 降「35%」，并新增 LTX-2.3-22B 与 X-VLA 实验（见论文归档）。
- 「80% 收益」按 FDCE 均值算：(12.63−8.91)/(12.63−7.97)≈0.80，与表一致；按中位数或 PSNR 算比例不同。
- 零动作 44.2 px vs 3.3 px 是单个 14B 样本，不是均值。
- 14B 结果无法用公开权重复现（仅放 2B）。

**对 wiki 的映射：** [paper-cd-lam](../../wiki/entities/paper-cd-lam.md)
