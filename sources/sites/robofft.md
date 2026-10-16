# RoboFFT 官方项目页

- **项目页：** https://student-of-holmes.github.io/RoboFFT/
- **论文：** [RoboFFT（arXiv:2609.32236）](https://arxiv.org/abs/2609.32236)
- **代码：** https://github.com/shshsunny/RoboFFT
- **模型 / 数据：** [Hugging Face](https://huggingface.co/shshsunny/RoboFFT/tree/master)；[ModelScope](https://modelscope.cn/models/shshsunny/RoboFFT/tree/master)（项目页称二者为相同 checkpoint 与数据镜像）
- **作者：** Yu Li、Shenghe Hu、Yuhan Wang、Yaoxiang Pu、Haotong Zhang、Yuanpei Chen、Yaodong Yang
- **机构：** 北京大学人工智能研究院、PKU-PsiBot 联合实验室、南京大学
- **会议：** CoRL 2026（项目页标注录用）
- **核查日期：** 2026-10-16
- **沉淀到 wiki：** [RoboFFT 实体页](../../wiki/entities/paper-robofft.md)

## 方法与公开结果摘录

RoboFFT 对 rollout 动作块沿生成策略的前向过程加噪，以加权 denoising / flow-matching loss 近似动作负对数似然；新旧策略的 loss 差形成用于 PPO clip 更新的 surrogate ratio。RoboFFT-F 对应 flow 策略，RoboFFT-D 对应 diffusion 策略。

项目页报告仿真训练效率：RoboFFT-F 相较 ReinFlow 平均约 1.53×，RoboFFT-D 在 Robomimic state-input 任务上相较 DPPO 约 38.0×。其真机 RL 页面给出四任务预训练成功次数平均 45.0% → online RoboFFT 93.8%；数字为官网展示值，比较时应回到论文核对试验预算与统计口径。

## 开源状态

- **部分开源（截至 2026-10-16）：** 项目页链接到公开 [shshsunny/RoboFFT](https://github.com/shshsunny/RoboFFT) 仓库。当前 README 说明 release 含 Robomimic 仿真的 RoboFFT-F（flow）与 RoboFFT-D（diffusion）训练/评测代码；公开 HF 与 ModelScope 镜像提供 checkpoint 与数据。
- **未随此 release 发布：** README 明确 VLA 与真实机器人实验代码不包含在当前版本中。自动下载所需的部分官方资产 URL 仍需填写，因此复现前需按 README 检查数据与 checkpoint 路径。
- **互指：** [论文来源](../papers/robofft_arxiv_2609_32236.md) · [代码来源](../repos/robofft.md)
