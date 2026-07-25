# Bavaria Direct Winding Scheme Calculator（Bewicklungsrechner XL）

> 来源归档

- **标题：** Winding Scheme Calculator / Bewicklungsrechner XL
- **类型：** repo（单页 Web 应用源码，standalone 离线包）
- **作者：** Felix Niessen（felix.niessen@googlemail.com，caendle.de），2010
- **上游基础：** POWERCROCO.DE 的「普通」绕组计算器，(C) 2004 overclocker_2001（oc2k1）与 DrM
- **致谢（源码头注释）：** Friedhelm S. 提供绕组系数（Wickelfaktor）公式与推导说明
- **托管方：** Bavaria Direct（南非），`https://www.bavaria-direct.co.za/scheme/calculator/`
- **许可：** GPL v3 or later（源码头部明确声明）
- **入库日期：** 2026-07-25
- **一句话说明：** 纯前端 JS 工具，输入槽数/极数即输出三相 BLDC 绕组排布字符串、齿槽定位次数、全极数谱的绕组系数表，并用 Canvas 画出定子绕线与星/角接线图。
- **开源状态：** **已开源**（GPLv3 源码随页面明文分发；无 GitHub 仓库，靠站点与论坛离线包流传）
- **站点状态：** **已下线**。截至入库日 `bavaria-direct.co.za` DNS 解析失败；SimpleFOC 社区讨论串确认工具不再可访问，标准包由 RCGroups「bavaria-direct going off-line」帖分发。
- **沉淀到 wiki：** [bavaria-direct-winding-calculator](../../wiki/entities/bavaria-direct-winding-calculator.md)

---

## 归档物（本次 ingest 的 standalone 包）

| 文件 | 行数 | 作用 |
|------|------|------|
| `winding_calc.shtml` | 63 | 宿主页面；只提供 `<div id="jsContainer">` 挂载点 |
| `bewicklungsrechner_xl_script.js` | 2005 | **全部逻辑**：排布生成、绕组系数 DFT、Canvas 绘图、中英双语 |
| `bewicklungsrechner_xl_style.css` | 146 | 计算器组件样式 |
| `default.css` | 92 | 站点通用样式 |

零依赖：无框架、无构建、无网络请求，`file://` 直接可跑。

## 版本核对

与 Wayback Machine 2014-05-28 抓取的线上副本
（`web.archive.org/web/20140528131642id_/http://bavaria-direct.co.za/scheme/calculator/bewicklungsrechner_xl_script.js`）
逐行 diff，仅 **3 处** 差异，全为外观/文案：

1. `lang['schritt_schritt_en']`：`Step by step` → `Winding animation`
2. `stator.fillStyle` 赋值位置调整两处 → 定子齿由近黑 `#111` 改为灰 `rgb(186,186,186)`

即：standalone 包是 Bavaria Direct 线上部署版的**后期修订**，算法与 2010 年 GPLv3 原版一致。

## 数值核验（本次 ingest 时做的复算）

把 `berechnen()` 的相带分配与 `WF_FFT()` 的 DFT 端口到 Python 重算，对照电机学教科书值：

| 槽/极 | 层数 | 源码输出 | 教科书 k_w |
|-------|------|----------|-----------|
| 12/10、12/14 | 双层集中 | 0.93301 | 0.933 |
| 9/8、9/10 | 双层集中 | 0.94521 | 0.945 |
| 12/10 | 单层集中 | 0.96593 | 0.966 |
| 24/22 | 双层集中 | 0.94947 | 0.949 |
| 12/8、6/4、3/2 | 双层集中 | 0.86603 | 0.866 |
| 36/4（q=3） | 分布整距 | 0.95980 | k_d = 0.9598 |
| 24/4（q=2） | 分布整距 | 0.96593 | k_d = 0.9659 |
| 24/4 短距 1/2/3 槽 | 分布短距 | 0.93301 / 0.83652 / 0.68301 | k_d·k_p 同值 |

槽口因子与斜槽因子公式亦与教科书 `sin(x)/x` 形式逐位吻合。结论：**该工具的绕组系数不是经验拟合，是可复现的解析/离散结果**。

## 对 wiki 的映射

- [Bavaria Direct 绕组方案计算器](../../wiki/entities/bavaria-direct-winding-calculator.md) — 源码原理详解（相带分配、电流负荷 DFT、平衡性判据、绘图）
- [开源力矩电机电磁设计完整度对比](../../wiki/comparisons/open-source-torque-motor-em-design.md) — 归位到「先定槽极绕组，再进 FEM」的工具链前置环节
- [FEMM-FOC-Simulation](../../wiki/entities/femm-foc-simulation.md) — 下游：拿到排布字符串后在 FEMM 里配绕组方向
- [力矩电机纵深路线 Stage 2](../../roadmap/depth-torque-motor-design.md) — 电磁与热设计阶段的选槽极工具
