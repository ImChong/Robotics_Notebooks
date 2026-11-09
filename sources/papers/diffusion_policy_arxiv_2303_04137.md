# Diffusion Policy: Visuomotor Policy Learning via Action Diffusion（arXiv:2303.04137）

> 来源归档（ingest）

- **标题：** Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
- **短名：** Diffusion Policy
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2303.04137>
- **PDF：** <https://arxiv.org/pdf/2303.04137>
- **项目页：** <https://diffusion-policy.cs.columbia.edu/>
- **代码：** <https://github.com/real-stanford/diffusion_policy>
- **机构：** 哥伦比亚大学；麻省理工
- **入库日期：** 2026-09-02
- **索引来源：** [具身智能研究室 VLA/WM 阅读路线](../blogs/wechat_embodied_ai_lab_vla_wm_reading_roadmap_2026-09-02.md)
- **一句话说明：** 用 DDPM 去噪生成动作 chunk；表达多峰 visuomotor 分布；π₀ 等 VLA 的动作头前身。

## 开源状态（步骤 2.5，2026-09-02）

- **已开源**：`real-stanford/diffusion_policy` + 项目页。

## 核心摘录（面向 wiki 编译）

- 训练：对专家动作加噪，网络预测噪声；推理：从高斯噪声逐步去噪得到动作序列。
- 相对 GMM/VAE 更适合多峰动作；去噪步数是速度–质量权衡。
- **滚动执行（receding horizon）：** 策略预测长度 \(T_p\) 的动作序列，部署时 **只执行前 \(T_e\) 步**（\(T_e < T_p\)），再用新观测 **重新去噪** 下一段——论文明确将此作为核心控制设计之一（非逐步单动作回归）。
- **对 wiki 的映射：** [paper-diffusion-policy](../../wiki/entities/paper-diffusion-policy.md)；方法页 [diffusion-policy](../../wiki/methods/diffusion-policy.md)；概念页 [receding-horizon-policy-execution](../../wiki/concepts/receding-horizon-policy-execution.md)

## 当前提炼状态

- [x] 项目页/仓库已交叉核查
- [x] wiki 映射：`wiki/entities/paper-diffusion-policy.md` 新建

## 2026-11-09 原文全文复核

- **核对范围：** 2026-11-09 通读 arXiv:2303.04137v5 正文 §1–10、实验 Table 1–6、Appendix A–D 的实现超参/消融/真机设置；未重跑作者训练脚本。
- **通读：** 已完成

- 视觉 ResNet-18 可使用 spatial softmax + GroupNorm，训练噪声预测模型、推理从噪声迭代恢复动作序列，执行前缀并滚动重规划；提供 CNN/Transformer 两类时间模型。论文在 15 项任务的基准比较报告平均 **46.9%** 改进（论文定义的跨任务汇总，不代表每个任务均提升 46.9 个百分点）。
- **审计发现 1：** 正文 §5.2 脚注明确指出 **robomimic 本应 50 个初始化，因评测代码 bug 实际只用了 22 个**；不要在百科页宣称所有 robomimic 结果遵循 50 初始化。
- **审计发现 2：** §3.4 文字举例 DDIM 用 100 训练/10 推理可达到 NVIDIA 3080 上 0.1 s；Appendix A.4 写真机实验 **100 训练/16 推理**。两个数字对应论文不同位置的描述，复现时应按配置表明确，不强行合并成一个数。
- 真机 Push-T 对目标区域 IoU 与成功判据分别衡量，后续液体操作与双臂任务还包含不同观测/动作和终止标准。单步控制频率不能从去噪迭代数直接推算。
- **局限：** 多步去噪引入额外推理时延，动作监督质量和控制接口决定部署效果。

> “全文通读”仅指本篇官方原文的主要章节及原文附录，不等同逐篇审计所有参考论文、运行作者代码或复现实验。
