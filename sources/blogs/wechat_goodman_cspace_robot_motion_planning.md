# 从真实空间到 C-space：机器人运动规划的第一张地图

> 来源归档（blog / 微信公众号 · Modern Robotics 原理精读）

- **标题：** 从真实空间到 C-space：机器人运动规划的第一张地图
- **类型：** blog
- **作者：** 写个 goodMan（微信公众号）
- **原始链接：** http://mp.weixin.qq.com/s?__biz=Mzg2ODgxOTA1Mw==&mid=2247483809&idx=1&sn=040d35532ec20f08e3a7666de1a10bf9&chksm=cea7c8a3f9d041b511bed857ac77ec34659153ebf33d1f289388477cd121fd4fe9436580a6a4#rd
- **发表日期：** 2026-05-17
- **入库日期：** 2026-10-01
- **抓取方式：** Agent Reach v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`playwright==1.49.1`）；专辑页同会话 `data-link` 跳转（直连 CAPTCHA）
- **专栏专辑：** [Modern Robotics 原理精读](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=Mzg2ODgxOTA1Mw==&action=getalbum&album_id=4521219024549937157)（第 1 篇 / 10）
- **原始抓取落盘：** [`sources/raw/wechat_modern_robotics_album_4521219024549937157/01_mid2247483809/01_mid2247483809.md`](../sources/raw/wechat_modern_robotics_album_4521219024549937157/01_mid2247483809/01_mid2247483809.md)
- **一句话说明：** 从位形、自由度、Grübler 公式到完整/非完整约束，把运动规划的第一张地图画在 C-space 上。

## 核心摘录（归纳，非全文）

- 位形描述整机关节状态，不是仅末端位置；同末端可对应不同位形。
- C-space 维数 = 独立 dof；形状可为环面 $S^1\times S^1$ 等，与「画成正方形」的展开图区分。
- Grübler 公式：$F=6(l-1)-\sum(6-f_i)$（空间）用于开链/闭链 dof 计数。
- 完整约束降低 C-space 维数；Pfaffian 非完整约束限制瞬时速度但通常不降维（平面小车典型）。
- 任务空间 / 工作空间 / C-space 三者的映射与冗余度决定规划在何空间做搜索。

## 对 wiki 的映射

- [configuration-space](../../wiki/formalizations/configuration-space.md)
- [modern-robotics-wechat-principles-series](../../wiki/overview/modern-robotics-wechat-principles-series.md)

## 可信度与使用边界

- 科普精读专栏，公式与符号对齐 Lynch & Park *Modern Robotics*；严格证明以教材 PDF 为准（见 [Modern Robotics 实体](../../wiki/entities/modern-robotics-book.md)）。
- 无项目页/代码仓；步骤 2.5 不适用。
- 图在微信 CDN；知识页用公式与 Mermaid 复述主干。

## 当前提炼状态

- [x] 专辑同会话抓取与 raw 归档
- [x] 归纳摘要与 wiki 挂接
