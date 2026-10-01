# Modern Robotics 第三章（1）：平面内的刚体运动

> 来源归档（blog / 微信公众号 · Modern Robotics 原理精读）

- **标题：** Modern Robotics 第三章（1）：平面内的刚体运动
- **类型：** blog
- **作者：** 写个 goodMan（微信公众号）
- **原始链接：** http://mp.weixin.qq.com/s?__biz=Mzg2ODgxOTA1Mw==&mid=2247483820&idx=1&sn=6da5efd636e9ebf34b176ef4009ad8bf&chksm=cea7c8aef9d041b81726f4ead46d4299565b55ab2db054859cb130693806c82452f59b5ad747#rd
- **发表日期：** 2026-05-19
- **入库日期：** 2026-10-01
- **抓取方式：** Agent Reach v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`playwright==1.49.1`）；专辑页同会话 `data-link` 跳转（直连 CAPTCHA）
- **专栏专辑：** [Modern Robotics 原理精读](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=Mzg2ODgxOTA1Mw==&action=getalbum&album_id=4521219024549937157)（第 2 篇 / 10）
- **原始抓取落盘：** [`sources/raw/wechat_modern_robotics_album_4521219024549937157/02_mid2247483820/02_mid2247483820.md`](../sources/raw/wechat_modern_robotics_album_4521219024549937157/02_mid2247483820/02_mid2247483820.md)
- **一句话说明：** Modern Robotics 第 3 章（1）：向量与参考系、平面 SE(2) 刚体运动与旋转矩阵下标规则。

## 核心摘录（归纳，非全文）

- 几何向量与坐标列向量分离；同一向量在不同基下坐标不同。
- 平面刚体位形 $(p_x,p_y,\theta)$，旋转 $R\in SO(2)$，$p'=Rp+t$。
- 下标消去规则：$R_{ac}=R_{ab}R_{bc}$，避免混用同一字母表示不同参考系。
- 为三维 SO(3)/SE(3)、twist 与齐次矩阵铺垫。

## 对 wiki 的映射

- [lie-group-rigid-body-motions](../../wiki/formalizations/lie-group-rigid-body-motions.md)
- [modern-robotics-wechat-principles-series](../../wiki/overview/modern-robotics-wechat-principles-series.md)

## 可信度与使用边界

- 科普精读专栏，公式与符号对齐 Lynch & Park *Modern Robotics*；严格证明以教材 PDF 为准（见 [Modern Robotics 实体](../../wiki/entities/modern-robotics-book.md)）。
- 无项目页/代码仓；步骤 2.5 不适用。
- 图在微信 CDN；知识页用公式与 Mermaid 复述主干。

## 当前提炼状态

- [x] 专辑同会话抓取与 raw 归档
- [x] 归纳摘要与 wiki 挂接
