# Modern Robotics 第三章（2）：旋转与角速度

> 来源归档（blog / 微信公众号 · Modern Robotics 原理精读）

- **标题：** Modern Robotics 第三章（2）：旋转与角速度
- **类型：** blog
- **作者：** 写个 goodMan（微信公众号）
- **原始链接：** http://mp.weixin.qq.com/s?__biz=Mzg2ODgxOTA1Mw==&mid=2247483845&idx=1&sn=1a17f0564509a453d0344b887a256894&chksm=cea7c8c7f9d041d199ec2c0a5b41a1d39e9c732caf97eb10cbd44c94599ab80a1dd67ac83f55#rd
- **发表日期：** 2026-05-23
- **入库日期：** 2026-10-01
- **抓取方式：** Agent Reach v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`playwright==1.49.1`）；专辑页同会话 `data-link` 跳转（直连 CAPTCHA）
- **专栏专辑：** [Modern Robotics 原理精读](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=Mzg2ODgxOTA1Mw==&action=getalbum&album_id=4521219024549937157)（第 3 篇 / 10）
- **原始抓取落盘：** [`sources/raw/wechat_modern_robotics_album_4521219024549937157/03_mid2247483845/03_mid2247483845.md`](../sources/raw/wechat_modern_robotics_album_4521219024549937157/03_mid2247483845/03_mid2247483845.md)
- **一句话说明：** 第 3 章（2）：SO(3) 旋转矩阵、角速度、指数坐标 $\omega$ 与矩阵对数。

## 核心摘录（归纳，非全文）

- 合法旋转矩阵：正交且 $\det R=1$；姿态与位置分开讨论。
- 角速度 $\omega$ 与 $\dot R = [\omega]_\times R$（空间）/ 体坐标形式对照。
- 有限旋转 = 绕轴 $\hat\omega$ 转 $\theta$；指数坐标 $\omega=\hat\omega\theta$。
- $R=\exp([\omega]_\times)$ 与 $\log R$ 取回轴角；为 se(3) 指数映射预热。

## 对 wiki 的映射

- [lie-group-rigid-body-motions](../../wiki/formalizations/lie-group-rigid-body-motions.md)
- [se3-representation](../../wiki/formalizations/se3-representation.md)
- [unit-quaternion-so3](../../wiki/formalizations/unit-quaternion-so3.md)

## 可信度与使用边界

- 科普精读专栏，公式与符号对齐 Lynch & Park *Modern Robotics*；严格证明以教材 PDF 为准（见 [Modern Robotics 实体](../../wiki/entities/modern-robotics-book.md)）。
- 无项目页/代码仓；步骤 2.5 不适用。
- 图在微信 CDN；知识页用公式与 Mermaid 复述主干。

## 当前提炼状态

- [x] 专辑同会话抓取与 raw 归档
- [x] 归纳摘要与 wiki 挂接
