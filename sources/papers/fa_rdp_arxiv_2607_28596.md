# FA-RDP: A Frequency-Adaptive Reactive Diffusion Policy for Contact-Rich Manipulation（arXiv:2607.28596）

> 来源归档（ingest）

- **标题：** FA-RDP: A Frequency-Adaptive Reactive Diffusion Policy for Contact-Rich Manipulation
- **缩写：** **FA-RDP**
- **类型：** paper / frequency-adaptive visual-force diffusion policy
- **arXiv：** <https://arxiv.org/abs/2607.28596>
- **PDF：** <https://arxiv.org/pdf/2607.28596>
- **项目页：** <https://fa-rdp.github.io/>
- **发表日期：** 2026-07-30（arXiv preprint）
- **作者：** Lifeng Zhuo*, Wendi Chen*, Han Xue, Shirun Tang, Jun Lv, Cewu Lu†, Chuan Wen†（* equal；† corresponding）
- **机构：** 上海交通大学（SJTU）；上海创智学院（Shanghai Innovation Institute）；诺玛矩阵（Noematrix）
- **入库日期：** 2026-08-02
- **一句话说明：** 接触前低频率多步扩散保多模态，接触后由多模态指示器切到 30 Hz 流形一致性蒸馏一步采样；Flexiv 三任务真机平均 **81.7%** SR。

## 核心论文摘录（MVP）

### 1) 问题：固定推理频率被迫折中（Abstract / §I）

- **链接：** <https://arxiv.org/abs/2607.28596>
- **核心贡献：** 接触丰富操作中，接触前需保留多条合法接近轨迹（多模态），接触后几何/力约束收窄且需快速力反馈。标准扩散策略固定频率：低频多步保模态但反应慢；高频反应快但易塌缩模态。FA-RDP 用 **共享多频率视觉–力 Transformer + 学习多模态指示器 + MCD 蒸馏** 按阶段切换。
- **对 wiki 的映射：**
  - [FA-RDP 论文实体](../../wiki/entities/paper-fa-rdp.md)
  - [Diffusion Policy](../../wiki/methods/diffusion-policy.md)
  - [Contact-Rich Manipulation](../../wiki/concepts/contact-rich-manipulation.md)

### 2) 多频率骨干 + 多模态指示器（§III）

- **核心贡献：**
  - 同一网络经 frequency-aware 位置编码同时预测 **10 Hz（H=16）** 与 **30 Hz（H=48）** 动作块（同 1.6 s 视界）。
  - 慢上下文来自视觉；快上下文刷新力/力矩 token。
  - 从慢 token 算 **multimodality indicator**，阈值门控：高歧义 → 多步 DDIM 低频采样；歧义下降 → 高频蒸馏采样。
- **对 wiki 的映射：**
  - [Hybrid Force-Position Control](../../wiki/concepts/hybrid-force-position-control.md)
  - [接触力旋量闭环 Query](../../wiki/queries/contact-wrench-closed-loop.md)

### 3) Manifold Consistency Distillation（MCD）（§III-C）

- **核心贡献：** 将高频采样器蒸馏为一步：网络直接预测 **动作流形上的 action chunk**，保留 DDPM 残差监督（MCD + SRL）；相对 epsilon/score/velocity 蒸馏更稳。部署与 ImplicitRDP 一致的 cache/consistent inference（Alg.1）；另共享 **100 Hz** 力补偿命令层 \(p_{\mathrm{cmd}}=p_{\mathrm{policy}}-\lambda f_{\mathrm{ext}}\)。
- **对 wiki 的映射：**
  - [扩散模型](../../wiki/concepts/diffusion-model.md)
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)

### 4) 真机评测：三任务 · Table I/II（§V）

- **核心贡献：**
  - 平台：Flexiv Rizon 4s；腕部 iPhone + 第三人称 USB；每任务 60 示教 / 20 评测。
  - 任务：Dual Box Flipping、Dual Switch Toggling、Dual Button Pressing（前方挡块制造多接近模态）。
  - **Table I Avg：** FA-RDP **81.7%**（14/18/17）vs ImplicitRDP 51.7%、RDP 35.0%、DP 10.0%、Regression+Force 20.0%。
  - **Table II：** 仅高频蒸馏 alone 61.7% → 指示器切换后 81.7%；Fig.8 显示 FA-RDP 保留四向接近模态而 HF-distill alone 塌缩。
- **对 wiki 的映射：**
  - [Manipulation](../../wiki/tasks/manipulation.md)
  - [OmniTacTune](../../wiki/entities/paper-omnitactune-tactile-residual-adaptation.md)（接触丰富适应对照）

## 开源状态（2026-08-02 项目页核查）

- **宣称将开源 / 训练代码未发布：** 项目页 Code 按钮为 `href="#"`，文案 **「Code (coming soon)」**；正文写 “Code will be made publicly available”。
- GitHub [`zhuolifeng/FA-RDP`](https://github.com/zhuolifeng/FA-RDP) **仅为站点源**（`index.html` / `root.pdf` / `figs`）；`releases/v1.0` 含对比视频，**无可辨识训练/推理入口**。
- 归档：[`sources/sites/fa-rdp-github-io.md`](../sites/fa-rdp-github-io.md)。

## 对 wiki 的映射（汇总）

- 沉淀实体页：[`wiki/entities/paper-fa-rdp.md`](../../wiki/entities/paper-fa-rdp.md)
- 交叉升级：
  - [Diffusion Policy](../../wiki/methods/diffusion-policy.md)
  - [Contact-Rich Manipulation](../../wiki/concepts/contact-rich-manipulation.md)
  - [Manipulation](../../wiki/tasks/manipulation.md)
  - [接触力旋量闭环 Query](../../wiki/queries/contact-wrench-closed-loop.md)

## 推荐继续阅读

- 项目页视频与结果表：<https://fa-rdp.github.io/>
- 对照基线原文：Reactive Diffusion Policy / ImplicitRDP（文中 [3][4]）
