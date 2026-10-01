# 刚体绕 z 轴转，螺旋轴却不一定是 z 轴？

> 来源归档（blog / 微信公众号 · Modern Robotics 原理精读）

- **标题：** 刚体绕 z 轴转，螺旋轴却不一定是 z 轴？
- **类型：** blog
- **作者：** 写个 goodMan（微信公众号）
- **原始链接：** http://mp.weixin.qq.com/s?__biz=Mzg2ODgxOTA1Mw==&mid=2247483947&idx=1&sn=0272154cf0687557d29bf40891237eeb&chksm=cea7cb29f9d0423fcf065da9a6d9193b8bcf08ab577c038e65cae0d886470f68af21687be248#rd
- **发表日期：** 2026-06-02
- **入库日期：** 2026-10-01
- **抓取方式：** Agent Reach v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`playwright==1.49.1`）；专辑页同会话 `data-link` 跳转（直连 CAPTCHA）
- **专栏专辑：** [Modern Robotics 原理精读](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=Mzg2ODgxOTA1Mw==&action=getalbum&album_id=4521219024549937157)（第 7 篇 / 10）
- **原始抓取落盘：** [`sources/raw/wechat_modern_robotics_album_4521219024549937157/07_mid2247483947/07_mid2247483947.md`](../sources/raw/wechat_modern_robotics_album_4521219024549937157/07_mid2247483947/07_mid2247483947.md)
- **一句话说明：** 关节轴 $\hat\omega$ 与螺旋轴 $\mathcal{S}=(\hat\omega,h)$ 不必重合：平移分量来自 $q\times\hat\omega$。

## 核心摘录（归纳，非全文）

- 绕 $z$ 转 90° 且带平面平移时，瞬时 screw 轴一般不是 $z$。
- 螺旋运动 = 绕空间某轴匀速转 + 沿轴匀速平移；轴可随位形变。
- 区分「关节几何轴」与「当前运动螺旋轴」，避免 PoE 建模张冠李戴。

## 对 wiki 的映射

- [spatial-twist-wrench-poe](../../wiki/formalizations/spatial-twist-wrench-poe.md)

## 可信度与使用边界

- 科普精读专栏，公式与符号对齐 Lynch & Park *Modern Robotics*；严格证明以教材 PDF 为准（见 [Modern Robotics 实体](../../wiki/entities/modern-robotics-book.md)）。
- 无项目页/代码仓；步骤 2.5 不适用。
- 图在微信 CDN；知识页用公式与 Mermaid 复述主干。

## 当前提炼状态

- [x] 专辑同会话抓取与 raw 归档
- [x] 归纳摘要与 wiki 挂接
