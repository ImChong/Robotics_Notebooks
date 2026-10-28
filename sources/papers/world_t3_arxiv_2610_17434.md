# World-T³（arXiv:2610.17434）

> 来源归档（paper）

- **标题：** World-T3: Test-Time Training as Persistent Memory for Real-Time Video World Models
- **作者：** Junyi Wu、Zekun Li 等
- **机构：** Meta Reality Labs、伊利诺伊大学芝加哥分校、布朗大学
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.17434>
- **项目页：** <https://adreamwu.github.io/World-T3/>
- **提交日期：** 2026-10-12
- **入库日期：** 2026-10-28
- **一句话说明：** 以测试时训练的快速权重保存历史视觉特征，并按相机位姿读取，以支持长时视频生成中的场景重访。

## 核心摘录（论文摘要及项目页）

- 用递归最小二乘在线更新位姿索引记忆；存储量和每步计算量相对于生成长度为常数。
- 先构建相机条件教师，再训练带记忆的自回归教师，最后用双教师分布匹配蒸馏至四步生成。
- 长时重访基准中，重访 PSNR 为 18.31 dB，最强基线为 13.52 dB；旋转误差为 1.075°，基线为 3.344°。
- 单张 NVIDIA B200 上以 1280×704 生成达到 20.6 FPS；该数字包含在线记忆更新。

## 开放状态

截至 2026-10-28，项目页未列代码、权重或数据下载链接；无法依据项目页确认公开可复现资源。

## 对 wiki 的映射

- [World-T³](../../wiki/entities/paper-world-t3.md)
- [生成式世界模型](../../wiki/methods/generative-world-models.md)
