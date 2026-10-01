# 为什么机器人学要用齐次变换矩阵描述刚体位姿？

> 来源归档（blog / 微信公众号 · Modern Robotics 原理精读）

- **标题：** 为什么机器人学要用齐次变换矩阵描述刚体位姿？
- **类型：** blog
- **作者：** 写个 goodMan（微信公众号）
- **原始链接：** http://mp.weixin.qq.com/s?__biz=Mzg2ODgxOTA1Mw==&mid=2247483879&idx=1&sn=f4d4ce4f59e84e1c80de358787dd0781&chksm=cea7c8e5f9d041f318f936d3a37ede7d3fb1e74ed8bcd56ff59e2e7c16fe43e1d1656c18b45b#rd
- **发表日期：** 2026-05-27
- **入库日期：** 2026-10-01
- **抓取方式：** Agent Reach v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`playwright==1.49.1`）；专辑页同会话 `data-link` 跳转（直连 CAPTCHA）
- **专栏专辑：** [Modern Robotics 原理精读](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=Mzg2ODgxOTA1Mw==&action=getalbum&album_id=4521219024549937157)（第 4 篇 / 10）
- **原始抓取落盘：** [`sources/raw/wechat_modern_robotics_album_4521219024549937157/04_mid2247483879/04_mid2247483879.md`](../sources/raw/wechat_modern_robotics_album_4521219024549937157/04_mid2247483879/04_mid2247483879.md)
- **一句话说明：** 用 $4\times4$ 齐次变换把 SE(3) 上的复合刚体运动统一成矩阵乘法。

## 核心摘录（归纳，非全文）

- 从 $SE(3)=\{(R,p)\}$ 到 $T\in\mathbb{R}^{4\times4}$ 的嵌入与分块结构。
- 点 $(x,y,z,1)$ 与方向 $(x,y,z,0)$ 区分平移敏感/不敏感。
- $T_{ac}=T_{ab}T_{bc}$ 链式；Example 3.19 强调用下标管理变换树。
- 与深蓝专栏齐次坐标文互补，推导对齐 *Modern Robotics* 符号。

## 对 wiki 的映射

- [homogeneous-coordinates-transform](../../wiki/formalizations/homogeneous-coordinates-transform.md)
- [se3-representation](../../wiki/formalizations/se3-representation.md)

## 可信度与使用边界

- 科普精读专栏，公式与符号对齐 Lynch & Park *Modern Robotics*；严格证明以教材 PDF 为准（见 [Modern Robotics 实体](../../wiki/entities/modern-robotics-book.md)）。
- 无项目页/代码仓；步骤 2.5 不适用。
- 图在微信 CDN；知识页用公式与 Mermaid 复述主干。

## 当前提炼状态

- [x] 专辑同会话抓取与 raw 归档
- [x] 归纳摘要与 wiki 挂接
