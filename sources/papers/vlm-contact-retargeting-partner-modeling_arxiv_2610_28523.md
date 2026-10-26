# VLM 接触重定向（arXiv:2610.28523）

> 来源归档（paper）

- **标题：** VLM-Guided Contact Retargeting and Generative Partner Modeling for Humanoid-Human Interaction
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.28523>
- **PDF：** <https://arxiv.org/pdf/2610.28523>
- **作者：** Yuto Shibata, Kotaro Amaya, Kashu Yamazaki, Lalit Jayanti, Yoshimitsu Aoki, Mariko Isogawa, Katerina Fragkiadaki
- **arXiv 评论栏：** CoRL2026, project page: this https URL
- **外链：** <https://yutoshibata07.github.io/REACT/>
- **入库日期：** 2026-10-26
- **一句话说明：** 面向人形—人交互的 VLM 引导接触重定向与生成式伙伴建模。

## 开源状态

- arXiv 摘要页/评论给出项目页或外链（https://yutoshibata07.github.io/REACT/）；未发现代码仓库链接（2026-10-26，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.19-10.23前沿论文动态】人形／四足39篇](../blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-19_23.md)
2. **周报所述问题：** 近距离人机接触会造成严重遮挡，使人体姿态估计失效；人类示范数据本身也可能因互相遮挡而产生错误。
3. **周报所述创新点：** 先由 VLM 识别语义接触并进行接触保持重定向，再将可访问伙伴真值的特权教师蒸馏为仅使用机载深度的学生；通过生成式伙伴状态预测处理视野外和遮挡人体。
4. **arXiv 摘要（英文节选）：** Humanoid-human interaction requires robots to operate under severe partial observability: during close physical contact, the human partner is occluded, truncated, or outside the robot's field of view, causing pose estimation to fail and often necessitating external tracking or privileged human state. At the same time, heavy mutual occlusions introduce errors into the human-human motion-capture data used for learning. We present REACT, a framework for learning deployable humanoid-human interaction policies from human-human demonstrations.

**对 wiki 的映射**

- [paper-vlm-contact-retargeting-partner-modeling](../../wiki/entities/paper-vlm-contact-retargeting-partner-modeling.md)
