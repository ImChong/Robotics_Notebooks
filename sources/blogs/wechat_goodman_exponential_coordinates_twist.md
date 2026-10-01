# 刚体运动的指数坐标：螺旋轴和位形如何互相转换？

> 来源归档（blog / 微信公众号 · Modern Robotics 原理精读）

- **标题：** 刚体运动的指数坐标：螺旋轴和位形如何互相转换？
- **类型：** blog
- **作者：** 写个 goodMan（微信公众号）
- **原始链接：** http://mp.weixin.qq.com/s?__biz=Mzg2ODgxOTA1Mw==&mid=2247483971&idx=1&sn=38fb6d1b209c1f8465a80394d8f79b46&chksm=cea7cb41f9d04257eaeae8a4a0dcc86c5f356e133204d3483d48b0c332dc368b0514caa27461#rd
- **发表日期：** 2026-06-03
- **入库日期：** 2026-10-01
- **抓取方式：** Agent Reach v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`playwright==1.49.1`）；专辑页同会话 `data-link` 跳转（直连 CAPTCHA）
- **专栏专辑：** [Modern Robotics 原理精读](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=Mzg2ODgxOTA1Mw==&action=getalbum&album_id=4521219024549937157)（第 8 篇 / 10）
- **原始抓取落盘：** [`sources/raw/wechat_modern_robotics_album_4521219024549937157/08_mid2247483971/08_mid2247483971.md`](../sources/raw/wechat_modern_robotics_album_4521219024549937157/08_mid2247483971/08_mid2247483971.md)
- **一句话说明：** se(3) 指数映射：给定 screw 轴与位移/转角，$T=\exp([\mathcal{S}]\theta)$ 与矩阵对数互逆。

## 核心摘录（归纳，非全文）

- $\mathcal{S}\theta$ 为 twist 的指数坐标；$\theta$ 为沿 screw 的广义位移。
- 平面例 3.26 手算 $\exp$ / $\log$ 验证 $T$ 与 $(\mathcal{S},\theta)$ 双向转换。
- PoE 正运动学把每个关节写成 $\exp([\mathcal{S}_i]\theta_i)$ 的乘积。

## 对 wiki 的映射

- [spatial-twist-wrench-poe](../../wiki/formalizations/spatial-twist-wrench-poe.md)
- [lie-group-rigid-body-motions](../../wiki/formalizations/lie-group-rigid-body-motions.md)

## 可信度与使用边界

- 科普精读专栏，公式与符号对齐 Lynch & Park *Modern Robotics*；严格证明以教材 PDF 为准（见 [Modern Robotics 实体](../../wiki/entities/modern-robotics-book.md)）。
- 无项目页/代码仓；步骤 2.5 不适用。
- 图在微信 CDN；知识页用公式与 Mermaid 复述主干。

## 当前提炼状态

- [x] 专辑同会话抓取与 raw 归档
- [x] 归纳摘要与 wiki 挂接
