# 运动旋量的一个核心视角：刚体瞬时速度场的 6 维坐标

> 来源归档（blog / 微信公众号 · Modern Robotics 原理精读）

- **标题：** 运动旋量的一个核心视角：刚体瞬时速度场的 6 维坐标
- **类型：** blog
- **作者：** 写个 goodMan（微信公众号）
- **原始链接：** http://mp.weixin.qq.com/s?__biz=Mzg2ODgxOTA1Mw==&mid=2247483928&idx=1&sn=a40aeeb2dbc13f229bd24e89109e852a&chksm=cea7cb1af9d0420c38766db092c9185014260bbb55930681871761b43b6d53166657e7d998c7#rd
- **发表日期：** 2026-05-30
- **入库日期：** 2026-10-01
- **抓取方式：** Agent Reach v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`playwright==1.49.1`）；专辑页同会话 `data-link` 跳转（直连 CAPTCHA）
- **专栏专辑：** [Modern Robotics 原理精读](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=Mzg2ODgxOTA1Mw==&action=getalbum&album_id=4521219024549937157)（第 6 篇 / 10）
- **原始抓取落盘：** [`sources/raw/wechat_modern_robotics_album_4521219024549937157/06_mid2247483928/06_mid2247483928.md`](../sources/raw/wechat_modern_robotics_album_4521219024549937157/06_mid2247483928/06_mid2247483928.md)
- **一句话说明：** 从距离不变性推导刚体速度场，说明 twist 是速度场在 6 维李代数上的坐标。

## 核心摘录（归纳，非全文）

- 刚体上任意两点速度差由 $\omega\times$ 决定，形成统一速度场。
- 选择基点（空间/物体）得到不同 twist 坐标，描述同一物理运动。
- 为 screw 轴、指数坐标与 PoE 正运动学提供速度侧直觉。

## 对 wiki 的映射

- [spatial-twist-wrench-poe](../../wiki/formalizations/spatial-twist-wrench-poe.md)

## 可信度与使用边界

- 科普精读专栏，公式与符号对齐 Lynch & Park *Modern Robotics*；严格证明以教材 PDF 为准（见 [Modern Robotics 实体](../../wiki/entities/modern-robotics-book.md)）。
- 无项目页/代码仓；步骤 2.5 不适用。
- 图在微信 CDN；知识页用公式与 Mermaid 复述主干。

## 当前提炼状态

- [x] 专辑同会话抓取与 raw 归档
- [x] 归纳摘要与 wiki 挂接
