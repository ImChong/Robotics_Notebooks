# 彻底搞懂RRT与RRT*｜为什么随机撒点，就能帮机器人找到一条路？又靠什么把路越改越短？

> 来源归档（blog / 微信公众号）

- **标题：** 彻底搞懂RRT与RRT*｜为什么随机撒点，就能帮机器人找到一条路？又靠什么把路越改越短？
- **类型：** blog
- **作者：** 深蓝学院 / 深蓝AI（《算法详解》专栏；编辑阿豹、审核阿蓝）
- **原始链接：** https://mp.weixin.qq.com/s/EeLwGT_pBUoEnIlsQObYdg
- **发表日期：** 2026-11-04
- **入库日期：** 2026-11-04
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信内置 UA），正文 6172 字符；文中 6 张示意图 / 动图未取回
- **一句话说明：** 算法教程：从「高维配置空间无法枚举」引出 RRT（采样、最近节点、扩展、碰撞检测），解释 Voronoi 偏置与概率完备性，再讲 RRT* 的选父节点与 Rewiring 及渐近最优性，最后给出选型与部署要点。
- **说明：** 文章是教程，没有项目清单，只有 5 条参考资料；文中点名的算法 RRT、RRT*、RRT-Connect、Informed RRT*（以及一张 Dubins-RRT 示意图）各建一个方法节点（Dubins-RRT 仅图中出现，未单独建节点）。

## 文章结构

1. 配置空间与枚举的代价（6 关节每关节 20 个角度 ≈ 6400 万状态，作者强调只是示意）
2. RRT 的四步：Sample → Nearest → Steer → Collision Check
3. 为什么随机撒点有效：Voronoi 偏置；概率完备性；目标偏置（Goal Bias）
4. 普通 RRT 的局限：只管「有没有路」
5. RRT*：选更优父节点 + Rewiring；渐近最优性的准确表述
6. RRT vs RRT*、变体（RRT-Connect、Informed RRT*）：有限算力用于「发现」还是「改进」
7. 部署：状态空间、连续运动碰撞检测、运动约束（Dubins 曲线）、路径到轨迹

## 参考资料（文章列出）

1. LaValle, *Rapidly-Exploring Random Trees: A New Tool for Path Planning*, 1998 — <https://msl.cs.illinois.edu/~lavalle/papers/Lav98c.pdf>（本次下载成功，但 PDF 文本不可机读，论文细节未独立核对）
2. Karaman & Frazzoli, *Sampling-based Algorithms for Optimal Motion Planning*, IJRR 2011 — <https://arxiv.org/abs/1105.1186>（摘要页已核对）
3. OMPL RRTstar 类文档 — <https://ompl.kavrakilab.org/classompl_1_1geometric_1_1RRTstar.html>
4. PythonRobotics RRT 页 — <https://atsushisakai.github.io/PythonRobotics/modules/5_path_planning/rrt/rrt.html>
5. LaValle, RRT Publications and Background — <https://lavalle.pl/rrtpubs.html>

## 对 wiki 的映射

- 新增方法节点：[RRT](../../wiki/methods/rrt.md)、[RRT*](../../wiki/methods/rrt-star.md)、[RRT-Connect](../../wiki/methods/rrt-connect.md)、[Informed RRT*](../../wiki/methods/informed-rrt-star.md)
- 选型对照页：[RRT 家族选型对照](../../wiki/comparisons/rrt-family-variants.md)（2026-11-04 补；RRT-Connect 依 OMPL 类文档与 LaValle《Planning Algorithms》§5.5.3 核对）
