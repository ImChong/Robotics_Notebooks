# OnOff: Bridging Online and Offline Handwriting via Differentiable Physical Rendering（arXiv:2608.03198）

> 来源归档（ingest）

- **标题：** Bridging Online and Offline Handwriting via Differentiable Physical Rendering（项目页作 **OnOff**）
- **缩写 / 框架：** **OnOff**；可微物理笔刷渲染器 \(\mathcal{R}\)
- **类型：** paper / robotic-calligraphy / differentiable-rendering / handwriting
- **arXiv：** <https://arxiv.org/abs/2608.03198>（PDF：<https://arxiv.org/pdf/2608.03198>）
- **项目页：** <https://seonmip.github.io/onoff/>（归档见 [`sources/sites/onoff-handwriting.md`](../sites/onoff-handwriting.md)）
- **会议：** ECCV 2026（项目页）
- **作者：** Seonmi Park、Seunghyun Shin、Vihaan Misra、Dongmin Shin、Ukcheol Shin†、Jean Oh、Hae-Gon Jeon⋆
- **机构：** 光州科学技术院（GIST）；卡内基梅隆大学（CMU）；延世大学（Yonsei）；韩国能源技术大学（KENTECH）
- **入库日期：** 2026-08-15
- **一句话说明：** 用六参数可微物理笔刷把 online 笔迹轨迹与 offline 手写图像接到同一框架，覆盖 text-to-stroke、笔刷参数观测、渲染与 zero-shot 图像精修，并在 UFACTORY Lite 6 上直接执行轨迹做机器人书法。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-15）：** [seonmip.github.io/onoff](https://seonmip.github.io/onoff/) 提供方法图、配对数据示意、离线生成对照与 Lite 6 真机书法视频；**未列 GitHub / 权重 / 数据下载**。
- **结论：** **项目页已发布，代码待核实 / 未开源。** wiki 源码运行时序图标 **不适用**。

## 摘录 1：问题与主张

- **痛点：** online 保留时序但缺纹理；offline 像真图但丢掉 stroke order；缺连接运动学与像素外观的显式物理模型，也缺配对 trajectory–image 数据。
- **主张：** 六参数笔刷 \(\theta=\{w_{\mathrm{base}},k_{\mathrm{spread}},\rho_{\mathrm{ink}},\sigma_{\mathrm{sharp}},p_{\min},p_{\max}\}\) + 可微渲染，把已有 online 语料合成配对数据，再训统一框架。
- **四模块：** text-to-stroke 生成器；brush parameter observer；differentiable brush renderer；zero-shot diffusion image refiner。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-onoff-handwriting.md`](../../wiki/entities/paper-onoff-handwriting.md)。

## 摘录 2：方法栈

| 模块 | 要点 |
|------|------|
| **压力代理** | 无压感数据时用速度反比 \(p_t^{\mathrm{proxy}}\)，再指数平滑 |
| **足迹** | 宽度随压力扩张；\(\rho_{\mathrm{ink}}\) 控墨色，\(\sigma_{\mathrm{sharp}}\) 控边缘；逐步 alpha 用 max 合成 |
| **数据** | 由 IAM-OnDB / CASIA-OLHWDB 渲染合成 \(D_{\mathrm{syn}}\)，叠真实背景 |
| **真机** | Lite 6：\((x,y)\) 走轨迹，压力代理映射 \(z\)；按 \(\theta\) 选铅笔 / 马克笔等工具 |

## 摘录 3：实验

- 自建 On-Offline 配对集：Our Renderer FID **11.81** vs One-DM **52.26**（Table 3）。
- 挂到 DiffPen / One-DM 作 prerender 引导，IAM / CVL 上 FID、BFID 常降（Table 4）；HWD / CER 随骨干与噪声注入步数权衡。
- 真机：多种书写工具与词级 / 句级拼接演示（句级用固定空间偏移，非端到端长句生成）。

## 建议 wiki 动作

- 新建实体页；交叉 [可微仿真](../../wiki/concepts/differentiable-simulation.md)、[视觉伺服](../../wiki/methods/visual-servoing.md)。
- 不建重复方法页。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（项目页无代码）
