# php-parkour.github.io（PHP 项目页）

- **标题：** Perceptive Humanoid Parkour — 官方项目页
- **类型：** site / project-page
- **URL：** <https://php-parkour.github.io/>
- **浏览器交互演示：** <https://php-parkour.github.io/demo.html>（MuJoCo Web demo；W/A/D 移动与转向，W 持续按住可攀爬，Y 切换速度，SPACE 暂停，BACKSPACE 重置）
- **代码：** <https://github.com/amazon-far/php_parkour>（官方训练 / motion matching / MuJoCo sim2sim 源码；Apache-2.0；归档见 [`sources/repos/amazon-far-php-parkour.md`](../repos/amazon-far-php-parkour.md)）
- **入库日期：** 2026-05-31
- **配套论文：** [PHP（arXiv:2602.15827）](https://arxiv.org/abs/2602.15827) — 归档见 [`sources/papers/php_parkour_arxiv_2602_15827.md`](../papers/php_parkour_arxiv_2602_15827.md)
- **PDF 镜像：** <https://php-parkour.github.io/static/images/paper.pdf>

## 一句话摘要

Amazon FAR 等人提出的 **Perceptive Humanoid Parkour (PHP)** 官方站点：展示 RSS 2026 论文视频、**浏览器内 MuJoCo 跑酷 demo**（W/A/D 前进转向、Y 切换高低速、攀爬时需持续按 W），以及高墙攀、vault、长程多障碍与**实时障碍位移**适应等实机片段。

## 公开信息要点（开源状态更新：2026-10-10）

- **机构：** Amazon FAR、UC Berkeley、CMU、Stanford University（* equal；† FAR co-lead）。
- **会议标签：** RSS 2026。
- **外链：** arXiv Paper、Video、Code；代码现已发布于 [`amazon-far/php_parkour`](https://github.com/amazon-far/php_parkour)，许可为 Apache-2.0。
- **交互 demo 操作提示（demo.html）：**
  - W 前进，A/D 转向；攀爬由按住 W 触发；
  - Y 切换速度，SPACE 暂停，BACKSPACE 重置。
- **演示技能（页面归纳）：** 10–15 cm side jump、cat vault + dash vault、speed vault、**1.25 m 墙攀 + roll**、从 1.25 m 墙滚落、0.58–0.76 m 攀+step、连续 stepping、**障碍实时位移**下的闭环适应。

## 为何值得保留

- **论文 Fig. 1 级能力的非 PDF 证据：** 视频与交互 demo 比静态摘要更直观呈现「长程技能链 + 感知决策」。
- **双层复现入口：** 浏览器 demo 直观看到深度策略如何根据障碍完成技能链；公开仓库则提供数据生成、IsaacSim 训练/评估与 MuJoCo sim2sim 实际代码，二者用途不同。
- **开放边界明确：** 仓库附带 motion/terrain 示例数据及学生 ONNX 文件，但没有原始 .pt teacher/student checkpoint；完整训练仍需 Linux/NVIDIA + IsaacSim。

## 关联资料

- 官方仓库：[`sources/repos/amazon-far-php-parkour.md`](../repos/amazon-far-php-parkour.md)

- 论文归档：[`sources/papers/php_parkour_arxiv_2602_15827.md`](../papers/php_parkour_arxiv_2602_15827.md)
- 上游重定向：[`sources/papers/omniretarget_arxiv_2509_26633.md`](../papers/omniretarget_arxiv_2509_26633.md)（PHP 正文引用 [43]）
- 姊妹篇索引：[`humanoid_rl_stack_22_*.md`](../papers/humanoid_rl_stack_22_perceptive_humanoid_parkour_chaining_dynamic_hum.md)
