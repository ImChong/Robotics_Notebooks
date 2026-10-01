# 为什么机器人学要用运动旋量描述刚体速度？

> 来源归档（blog / 微信公众号 · Modern Robotics 原理精读）

- **标题：** 为什么机器人学要用运动旋量描述刚体速度？
- **类型：** blog
- **作者：** 写个 goodMan（微信公众号）
- **原始链接：** http://mp.weixin.qq.com/s?__biz=Mzg2ODgxOTA1Mw==&mid=2247483867&idx=1&sn=cb0452d6522ce3eb99666d2181492fe6&chksm=cea7c8d9f9d041cfa28864cd43d5531e9a1f8476ff488b283c19ba898d9945e8b9d599d0b743#rd
- **发表日期：** 2026-05-27
- **入库日期：** 2026-10-01
- **抓取方式：** Agent Reach v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`playwright==1.49.1`）；专辑页同会话 `data-link` 跳转（直连 CAPTCHA）
- **专栏专辑：** [Modern Robotics 原理精读](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=Mzg2ODgxOTA1Mw==&action=getalbum&album_id=4521219024549937157)（第 5 篇 / 10）
- **原始抓取落盘：** [`sources/raw/wechat_modern_robotics_album_4521219024549937157/05_mid2247483867/05_mid2247483867.md`](../sources/raw/wechat_modern_robotics_album_4521219024549937157/05_mid2247483867/05_mid2247483867.md)
- **一句话说明：** 运动旋量 $\mathcal{V}=[\omega;v]$：统一描述刚体瞬时速度，区分空间 twist 与物体 twist。

## 核心摘录（归纳，非全文）

- 时变齐次变换 $T(t)$ 导出 $[\mathcal{V}_s]=\dot T T^{-1}$ 与 $[\mathcal{V}_b]=T^{-1}\dot T$。
- $\omega\times p + v$ 给出刚体速度场；twist 是其 6 维坐标。
- 空间/物体 twist 通过 Adjoint 互转；小车例 3.23 对比两种写法。
- 螺旋轴 $(\hat\omega,h)$ 与 pitch 把 twist 与几何运动对应。

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
