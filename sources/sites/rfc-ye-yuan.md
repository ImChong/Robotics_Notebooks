# ye-yuan.com/rfc（Residual Force Control 项目页）

- **标题：** Residual Force Control for Agile Human Behavior Imitation and Extended Motion Synthesis
- **类型：** site / project-page
- **URL：** <https://www.ye-yuan.com/rfc>（当前 301 跳转至 <https://ye-yuan.com/rfc/>）
- **配套论文：** [Residual Force Control（arXiv:2006.07364）](https://arxiv.org/abs/2006.07364)，NeurIPS 2020
- **代码：** <https://github.com/Khrylx/RFC> — 归档见 [`sources/repos/rfc-residual-force-control.md`](../repos/rfc-residual-force-control.md)
- **入库日期：** 2026-07-28

## 一句话摘要

Ye Yuan、Kris Kitani（CMU）的 RFC 官方项目页：在动作空间中加入作用于人形根部的**外部残差力**，补偿人体 MoCap 与仿真角色之间的动力学失配，实现芭蕾舞（pirouette/arabesque/jeté）等高难动作模仿与 Human3.6M 大规模长序列动作合成；页面含摘要、视频与 BibTeX。

## 公开信息要点（截至入库日）

- **机构：** Carnegie Mellon University（Robotics Institute）。
- **页面板块：** 摘要、演示视频（双栏 gif/video）、Citation；代码链接指向 GitHub（Khrylx/RFC）。
- **核心演示：** 芭蕾舞三动作、backflip、cartwheel、side flip 等；dual-policy（kinematic + RFC）无限时动作生成。
- **警示（论文/页面一致）：** 残差力是**仿真特权**——真实机器人不存在凭空作用于骨盆的外力，适合仿真训练与动作生成。

## 为何值得保留

RFC 是 Residual 思想在**动作模仿/角色动画**分支的代表：残差不在关节动作空间而在**力空间**，与 DeepMimic 谱系直接对位；项目页是核查其代码开放（Khrylx/RFC，非商用许可）的一手依据。

## 对 wiki 的映射

- 实体页：[paper-rfc-residual-force-control](../../wiki/entities/paper-rfc-residual-force-control.md)
- 方法页：[residual-policy-learning](../../wiki/methods/residual-policy-learning.md)
