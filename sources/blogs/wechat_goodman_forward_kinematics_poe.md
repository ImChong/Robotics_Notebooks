# 机器人学中的正向运动学到底在算什么？

> 来源归档（blog / 微信公众号 · Modern Robotics 原理精读）

- **标题：** 机器人学中的正向运动学到底在算什么？
- **类型：** blog
- **作者：** 写个 goodMan（微信公众号）
- **原始链接：** http://mp.weixin.qq.com/s?__biz=Mzg2ODgxOTA1Mw==&mid=2247484173&idx=1&sn=574f1696b63a16277bf38fbd52d6d3bb&chksm=cea7ca0ff9d04319309494b8b1925d425a0690218c4f7b4b2064799047b5591f08fc4484c11c#rd
- **发表日期：** 2026-06-08
- **入库日期：** 2026-10-01
- **抓取方式：** Agent Reach v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`playwright==1.49.1`）；专辑页同会话 `data-link` 跳转（直连 CAPTCHA）
- **专栏专辑：** [Modern Robotics 原理精读](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=Mzg2ODgxOTA1Mw==&action=getalbum&album_id=4521219024549937157)（第 10 篇 / 10）
- **原始抓取落盘：** [`sources/raw/wechat_modern_robotics_album_4521219024549937157/10_mid2247484173/10_mid2247484173.md`](../sources/raw/wechat_modern_robotics_album_4521219024549937157/10_mid2247484173/10_mid2247484173.md)
- **一句话说明：** 正运动学 = 零位形 $M$ 与各关节 $\exp([\mathcal{S}_i]\theta_i)$ 的乘积；空间/物体 PoE 形式对照。

## 核心摘录（归纳，非全文）

- 平面 3R 与空间 3R 开链：先写零位 $M$，再列各关节 screw $\mathcal{S}_i$。
- 空间形式 $T=\exp([\mathcal{S}_1]\theta_1)\cdots\exp([\mathcal{S}_n]\theta_n)M$。
- 体坐标形式把指数乘在右侧；与 DH 连乘等价但 screw 来自几何。
- 常见错误：screw 轴未在零位形下表达、下标系不一致。

## 对 wiki 的映射

- [forward-kinematics](../../wiki/formalizations/forward-kinematics.md)
- [spatial-twist-wrench-poe](../../wiki/formalizations/spatial-twist-wrench-poe.md)

## 可信度与使用边界

- 科普精读专栏，公式与符号对齐 Lynch & Park *Modern Robotics*；严格证明以教材 PDF 为准（见 [Modern Robotics 实体](../../wiki/entities/modern-robotics-book.md)）。
- 无项目页/代码仓；步骤 2.5 不适用。
- 图在微信 CDN；知识页用公式与 Mermaid 复述主干。

## 当前提炼状态

- [x] 专辑同会话抓取与 raw 归档
- [x] 归纳摘要与 wiki 挂接
