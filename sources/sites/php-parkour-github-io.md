# php-parkour.github.io（PHP 项目页）

- **标题：** Perceptive Humanoid Parkour — 官方项目页
- **类型：** site / project-page
- **URL：** <https://php-parkour.github.io/>
- **浏览器演示：** <https://php-parkour.github.io/demo.html>（MuJoCo 全浏览器交互 demo，模型流式加载）
- **代码：** <https://github.com/amazon-far/php_parkour>（已开源；[仓库归档](../repos/php_parkour.md)）
- **数据与权重：** [官方 Releases](https://github.com/amazon-far/php_parkour/releases)，motion matching 数据库、五组预制训练示例、学生 ONNX 双模型
- **入库日期：** 2026-05-31
- **更新核查：** 2026-10-10
- **配套论文：** [PHP（arXiv:2602.15827）](https://arxiv.org/abs/2602.15827) — 归档见 [`sources/papers/php_parkour_arxiv_2602_15827.md`](../papers/php_parkour_arxiv_2602_15827.md)
- **PDF 镜像：** <https://php-parkour.github.io/static/images/paper.pdf>

## 一句话摘要

Amazon FAR 等人提出的 **Perceptive Humanoid Parkour (PHP)** 官方站点：展示 RSS 2026 论文视频、**浏览器内 MuJoCo 跑酷 demo**（W/A/D 前进转向、Y 切换高低速、攀爬时需持续按 W），以及高墙攀、vault、长程多障碍与**实时障碍位移**适应等实机片段。

## 公开信息要点（2026-10-10 更新）

- **机构：** Amazon FAR、UC Berkeley、CMU、Stanford University（* equal；† FAR co-lead）。
- **会议标签：** RSS 2026。
- **外链：** arXiv Paper、Video、Code 已指向 `amazon-far/php_parkour`、Related Research。
- **历史状态：** 2026-07-20 核查时仍为 *Coming Soon*；本次更新以当前项目页与官方仓库为准，不再将历史待发布状态当作现状。
- **开放边界：** 训练与推理代码、数据库和学生 ONNX 已公开；五组预制训练数据是示例子集，未提供现成教师 checkpoint。详见[仓库归档](../repos/php_parkour.md)。
- **交互 demo 操作提示（主页 / mobile）：**
  - 对齐障碍后**持续按住 W** 完成接近、攀爬、台上与下落全过程；
  - A/D 微调对准下一障碍；Y 切换低/高速；SPACE 暂停；BACKSPACE 或 Reload 重置。
- **演示技能（页面归纳）：** 10–15 cm side jump、cat vault + dash vault、speed vault、**1.25 m 墙攀 + roll**、从 1.25 m 墙滚落、0.58–0.76 m 攀+step、连续 stepping、**障碍实时位移**下的闭环适应。

## 为何值得保留

- **论文 Fig. 1 级能力的非 PDF 证据：** 视频与交互 demo 比静态摘要更直观呈现「长程技能链 + 感知决策」。
- **可复现入口：** 浏览器 demo 降低交互门槛；训练与本地 MuJoCo 路线另见官方仓库，不能把网页运行成功等同于训练或真机复现。
- **与 arXiv / 项目 PDF 三角互证：** 摘要、方法图与站点演示一致，便于 lint 时核对表述。

## 关联资料

- 官方实现归档：[`sources/repos/php_parkour.md`](../repos/php_parkour.md)
- 论文归档：[`sources/papers/php_parkour_arxiv_2602_15827.md`](../papers/php_parkour_arxiv_2602_15827.md)
- 上游重定向：[`sources/papers/omniretarget_arxiv_2509_26633.md`](../papers/omniretarget_arxiv_2509_26633.md)（PHP 正文引用 [43]）
- 姊妹篇索引：[`humanoid_rl_stack_22_*.md`](../papers/humanoid_rl_stack_22_perceptive_humanoid_parkour_chaining_dynamic_hum.md)
