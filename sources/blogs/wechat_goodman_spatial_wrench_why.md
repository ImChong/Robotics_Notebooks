# 为什么机器人学要用力旋量描述外力？

> 来源归档（blog / 微信公众号 · Modern Robotics 原理精读）

- **标题：** 为什么机器人学要用力旋量描述外力？
- **类型：** blog
- **作者：** 写个 goodMan（微信公众号）
- **原始链接：** http://mp.weixin.qq.com/s?__biz=Mzg2ODgxOTA1Mw==&mid=2247484007&idx=1&sn=069d2a1343e001029f1e52a19a071a8c&chksm=cea7cb65f9d0427380b5e087ed88461b6b90114894534139f86579c08c5c3701af7db1bd40cc#rd
- **发表日期：** 2026-06-04
- **入库日期：** 2026-10-01
- **抓取方式：** Agent Reach v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`playwright==1.49.1`）；专辑页同会话 `data-link` 跳转（直连 CAPTCHA）
- **专栏专辑：** [Modern Robotics 原理精读](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=Mzg2ODgxOTA1Mw==&action=getalbum&album_id=4521219024549937157)（第 9 篇 / 10）
- **原始抓取落盘：** [`sources/raw/wechat_modern_robotics_album_4521219024549937157/09_mid2247484007/09_mid2247484007.md`](../sources/raw/wechat_modern_robotics_album_4521219024549937157/09_mid2247484007/09_mid2247484007.md)
- **一句话说明：** 力旋量 $\mathcal{F}=[f;\tau]$ 把力与力矩合成 6 维量，与 twist 对偶且经 Adjoint 转置变换。

## 核心摘录（归纳，非全文）

- 同力不同作用点力矩不同；选参考点把 $(f,\tau)$ 打包成 wrench。
- 虚功 $\mathcal{F}^\top \mathcal{V}$ 为功率；与 twist 配对做功分析。
- wrench 坐标变换用 $\mathrm{Ad}^T$，与 twist 的 Adjoint 对偶。
- 六维力传感器读数即 spatial wrench（例 3.28）。

## 对 wiki 的映射

- [spatial-twist-wrench-poe](../../wiki/formalizations/spatial-twist-wrench-poe.md)
- [contact-wrench-cone](../../wiki/formalizations/contact-wrench-cone.md)

## 可信度与使用边界

- 科普精读专栏，公式与符号对齐 Lynch & Park *Modern Robotics*；严格证明以教材 PDF 为准（见 [Modern Robotics 实体](../../wiki/entities/modern-robotics-book.md)）。
- 无项目页/代码仓；步骤 2.5 不适用。
- 图在微信 CDN；知识页用公式与 Mermaid 复述主干。

## 当前提炼状态

- [x] 专辑同会话抓取与 raw 归档
- [x] 归纳摘要与 wiki 挂接
