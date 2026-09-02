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
- **对 wiki 的映射：** [paper-diffusion-policy](../../wiki/entities/paper-diffusion-policy.md)；方法页 [diffusion-policy](../../wiki/methods/diffusion-policy.md)

## 当前提炼状态

- [x] 项目页/仓库已交叉核查
- [x] wiki 映射：`wiki/entities/paper-diffusion-policy.md` 新建
