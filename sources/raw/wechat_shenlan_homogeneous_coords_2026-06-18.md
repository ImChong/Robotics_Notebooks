# 补全具身智能L0级必备知识：齐次坐标与齐次变换 — 抓取落盘

> Agent Reach + wechat-article-for-ai 抓取落盘（2026-06-18）

- **标题：** 补全具身智能L0级必备知识：齐次坐标与齐次变换
- **作者：** 深蓝具身智能
- **链接：** https://mp.weixin.qq.com/s/3vwaizPOgJKCwQ9e5LuKGA
- **抓取工具：** Agent Reach v1.5.0 + `~/.agent-reach/tools/wechat-article-for-ai`（Camoufox）
- **正文规模：** 约 13580 字符 / 22 图 / 4 代码块
- **归纳归档：** [`sources/blogs/wechat_shenlan_homogeneous_coordinates_transform.md`](../blogs/wechat_shenlan_homogeneous_coordinates_transform.md)

## 抓取说明

- 专辑页 `album_id=4525948187102363653` 可枚举 5 篇；本篇为第 5 篇（`sn=cb7a648181e8cb27dc0af12570297329`，短链 `3vwaizPOgJKCwQ9e5LuKGA`）。
- 用户提供的短链可稳定抓取；部分带 `chksm` 的长链在 Cloud IP 上易触发 CAPTCHA。

## 目录结构（文内）

1. 为何刚体运动需要「升维思考」
2. 通俗解读 + 数学定义：齐次坐标
3. 核心推导：齐次变换矩阵（SE(3) 标准形式）
4. 齐次变换五大核心特性
5. 具身智能全场景落地（机械臂 FK、视觉/SLAM、自动驾驶、端到端 se(3) 优化）

## 关键公式（摘录）

- 原始：$p' = Rp + t$
- 齐次：$\tilde p' = T\tilde p$，$T=\begin{bmatrix}R&t\\0&1\end{bmatrix}$
- 连乘：$T_{\text{total}} = T_n \cdots T_1$

完整 Markdown 含 YAML frontmatter 与代码块，抓取于 `/tmp/wechat_out/`（不入库）。
