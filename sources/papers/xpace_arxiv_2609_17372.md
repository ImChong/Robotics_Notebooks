# XPACE（arXiv:2609.17372）

> 来源归档（paper · arXiv:2609.17372v1 全文核对）

- **标题：** XPACE: Joint World and Action Modeling from Heterogeneous Experience
- **作者：** Jiacheng Wei\*§, Jerry Bai\*§, Xiaoyu Yue\*, Zidong Wang\*, Xiaoyang Guo\*†, Cheng Chen, Fanqi Pu, Fan Wu, Zhixu Yue, Yizhuo Li, Feng Qiu, Bo Liu, Yuying Ge, Hui Zhou, Chenyi Chen, Yixiao Ge‡（† Project lead；‡ Supervision）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.17372>（v1，2026-09-15）
- **HTML 全文：** <https://arxiv.org/html/2609.17372>
- **PDF：** <https://arxiv.org/pdf/2609.17372> · [项目页 PDF](https://xpeng-robotics.github.io/xpace/assets/papers/xpace.pdf)
- **项目页：** <https://xpeng-robotics.github.io/xpace/>
- **机构：** XPENG Robotics（小鹏机器人）
- **入库日期：** 2026-09-20
- **核对范围：** 通读 `arXiv:2609.17372v1` HTML 全文（§1–5、Table 1–4、Fig. 1–13）及官方项目页 <https://xpeng-robotics.github.io/xpace/>，核查 `xpeng-robotics` GitHub 组织无 XPACE 公开仓库/权重。未运行模型训练或真机实验。
- **通读：** 已完成
- **一句话说明：** 统一 WAM + world simulator：共享 causal Video Transformer backbone 从 5,000h 异构经验联合学视频与动作，Stage III 经 SGF 适配 simulator 并合成 deviation–recovery 轨迹以 8% 混合比 DAgger 微调 policy；IRON 人形真机验证人–机技能迁移与自改进增益。

## 开源状态（步骤 2.5，2026-09-20 / 2026-11-22 复核）

- **未开源：** 项目页仅链至 [xpeng-robotics](https://github.com/xpeng-robotics) 组织，**无 XPACE 专用仓库或公开权重**；详见 [`sources/sites/xpace-project.md`](../sites/xpace-project.md)。

## 全文核心摘录（arXiv:2609.17372v1）

1. **双模式非对称 MoT 架构（§3.2）：**
   - 共享因果 Video Transformer（anchor latent $z_0$ + 每 chunk 4 latent frames = 16 video frames，chunk 内双向注意力、跨 chunk 因果注意力）。
   - **Simulation 模式：** 输入历史视觉、预设 2D 动作骨架序列 $\kappa^m$（经冻结 video VAE 编码并经零初始化 patch embedding 以 token addition 注入）与相机位姿 $\mathbf{c}_{\mathrm{cam}}^m$，预测 $p(\mathbf{z}^m \mid z_0, \mathbf{z}^{<m}, \kappa^m, \mathbf{c}_{\mathrm{cam}}^m)$；Action Transformer 关闭且无语言条件。
   - **Video-Action 模式：** 15 个 Action Transformer block 分别读取 Video Transformer 第 16–30 层的空间稠密视觉特征（无空间池化），结合本体状态 $\mathbf{s}^m$（训练时以 0.5 概率随机掩码）与实施例特定线性输入/输出投影，预测 $N_a=16$ 步动作块；动作损失梯度回传更新共享 Video Transformer。
2. **真机异步部署（§3.2 Real-robot deployment）：** 流式 VAE 每 4 帧生成 1 个 latent frame；推理器异步生成 16 步动作块，发布端以约 **10 Hz** 做 **overlapping-horizon stitching**（保留已提交指令、裁剪重叠前缀并施加衰减修正平滑拼接）。
3. **5,000h 异构数据金字塔与多级字幕（§3.1）：** L1 无动作 egocentric 视频、L2 人手–腕视频–动作（共享运动学 schema）、L3 任务/外观对齐 bridge、L4 IRON 遥操作；**F 失败/低进度/恢复集仅用于 simulation 更新，绝不进 policy 模仿**。每条样本配原始任务指令、稠密描述、原子动作命令与 LLM 改写，训练期间采样权重从稠密描述逐步过渡到任务指令。
4. **三阶段训练 + Stage III 自改进分叉（§3.3–3.4）：**
   - **Stage I：** 无动作视频适配预训练 Video Transformer（Action 分支未初始化）。
   - **Stage II（Phase II-a/b/c）：** 50/50 采样 video–action（$\mathcal{L}_v+\mathcal{L}_a$）与 simulation（$\mathcal{L}_v$），数据源从 L1/L2 粗到细过渡到 L3/L4（II-b 引入 F 到 simulation；II-c 仍保留人类 replay 防止过拟合与遗忘）。
   - **Stage III-sim（SGF）：** 自 Phase II-c 复制并固定 8 步 ODE 采样器自回归 rollout（无梯度追踪），再带梯度重编码生成前缀做上下文感知 flow matching。
   - **Stage III-policy（DAgger 微调）：** 固定 SGF simulator 渲染偏离–恢复轨迹并做一致性过滤，以 **8% 合成 recovery + 92% Phase II-c 配方** 微调独立 policy 副本（<1 天）。
5. **定量结果（§4）：**
   - **仿真预测：** Token addition 优于 AdaLN/cross-attn/concat；SGF 在 193 帧上使 ID/OOD PSNR 提升 +0.89/+1.01 dB 并提速 3.00×/3.30×；最终 Phase II-c + SGF 在 97 帧 ID PSNR **19.73 dB**（Table 3）。
   - **离线动作 loss：** Stage I 满量预训练相对 $r=0$ 降 **12.5%**；human–robot co-training 相对 robot-only 降 **14.0%**（mid-train→robot-only FT 为 −8.5%）；robot-absent 但 human-dense 行为降 **22.7%**。
   - **IRON 真机（20 trials/task）：** 香蕉 pick-place（16/20, 80%）/ 倒水（10/20, 50%）/ 叠碗（15/20, 75%，L4 无示范）平均 **68.3%**（progress 0.84），优于 DreamZero **40.0%**（0.68）与 GR00T **6.7%**（0.36）；注意 XPACE 独享 Stage I 适配。Recovery DAgger 微调后（Banana / Pour / Cola handover）平均成功率 **61.7%→86.7%**（progress 0.81→0.93），其中倒水 **50%→95%**。

## 对 wiki 的映射

- [paper-xpace](../../wiki/entities/paper-xpace.md)
- [xpace-project.md](../sites/xpace-project.md) — 项目页定量表与 demo 索引
- [world-action-models](../../wiki/concepts/world-action-models.md) — WAM 概念族谱
- [generative-world-models](../../wiki/methods/generative-world-models.md) — 生成式世界模型方法层
