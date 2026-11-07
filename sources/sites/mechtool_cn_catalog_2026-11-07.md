# Mechtool 在线机械设计工具（站点目录归档）

> 来源归档（site / 在线机械设计资料与计算工具站）

- **标题：** Mechtool 在线机械设计工具
- **类型：** site
- **原始链接：** https://www.mechtool.cn/
- **入库日期：** 2026-11-07
- **抓取方式：** curl 取首页 HTML，并对站内全部 190 个链接逐个请求（返回 HTTP 200 的 186 个，已记录页面标题与描述）；另读了 `/thread/index.html`（公制、美制和英制螺纹）与 `/calculation/calculation_fastenercalculationtool.html`（紧固件计算工具）两页全文。
- **核对范围：** 2026-11-07 用 curl 取回全部 186 个 HTTP 200 页面的 HTML，提取可见文本（去掉各页重复的导航菜单）后，由 6 个阅读子任务把提取文本**从头到尾**各读一遍（短页 / 索引页 / 数据页 / 供应商页 4 份，`/calculation/` 60 页 1 份，`/formular/` 26 页 1 份；另有前一轮读过的顶层页与螺纹、紧固件页）；子任务笔记在 `/tmp/claude-0/wx/mech/todo/notes_*.md`。**口径（写在前面，请按此理解「已完成」）：** ① 读的是抽取后的可见文字；**图片（公式图、示意图、约 550 张）没看**；② **没有运行任何计算器，没有读计算 JS**；③ 纯数值大表（轴承基本尺寸、螺纹偏差 / 极限尺寸、弹簧标准表、键 / 花键 / 平带 / e-Y 插值表等）在抽取文本里行列结构已丢失或成「一格一行」数值流，**只读表头、表名、所依据标准和符号说明，数值行仅按结构检查，没有逐行核对**；④ 部分页面抽取后只有导航或目录，没有正文；含 `<`、`>` 的句子在抽取时被截断；⑤ 公式正确性只由子任务在代入数值时抽查，未系统核对。
- **通读：** 已完成
- **一句话说明：** 面向机械设计的中文资料导航 + 在线计算器站点：七大板块（机械设计基础资料、机械零部件设计、机械传动 / 润滑与密封、液压与气压传动、机械书籍、计算机辅助设计、机械资料），首页还挂出常用小工具。

## 目录结构（实测）

- 站内链接 190 个：186 个 HTTP 200；`/mechanicalmonograph/index.html` 返回 404；另有 `javascript:;` 与 `mailto:` 等非页面链接。
- 外链 25 个，其中 11 个是 `cn.fastencloud.com` 的紧固件小工具（见下表）。
- `/calculation/` 下 60 个计算 / 查询页（见下表）。

## 计算 / 查询页（`/calculation/`，标题取自页面）

| 路径 | 标题 |
|------|------|
| `/calculation/calculation_cablechainlengthcalculation.html` | 拖链长度计算 |
| `/calculation/calculation_hardnessconversiontool.html` | 硬度换算工具 |
| `/calculation/calculation_deflectionapp.html` | 结构梁计算器 |
| `/calculation/calculation_fastenercalculationtool.html` | 紧固件计算工具 |
| `/calculation/calculation_calculationofsteelweight.html` | 钢材重量计算 |
| `/calculation/calculation_calculationofmaterialweight.html` | 材料重量计算 |
| `/calculation/calculation_tolerances.html` | 公差与配合查询 |
| `/calculation/calculation_shapeandpositiontolerance.html` | 形状与位置公差查询 |
| `/calculation/calculation_boltconnection.html` | 螺栓连接设计校核 |
| `/calculation/calculation_keyconnection.html` | 键连接设计校核 |
| `/calculation/calculation_springs.html` | 弹簧设计 |
| `/calculation/calculation_linearbearingselection.html` | 直线轴承选型 |
| `/calculation/calculation_linearguideselection.html` | 直线导轨选型 |
| `/calculation/calculation_screwdrive.html` | 螺旋传动设计 |
| `/calculation/calculation_ballscrewdrive.html` | 滚珠丝杆副设计计算 |
| `/calculation/calculation_beltdrive.html` | 带传动设计 |
| `/calculation/calculation_chaindrive.html` | 链传动设计 |
| `/calculation/calculation_double-speedrollerchainselection.html` | 倍速滚子链选型 |
| `/calculation/calculation_geardrive.html` | 渐开线圆柱齿轮传动设计 |
| `/calculation/calculation_wormandwormwheeldrive.html` | 普通圆柱蜗杆传动设计 |
| `/calculation/calculation_camindexerselection.html` | 凸轮分度(割)器选型 |
| `/calculation/calculation_hydraulicdrivesystemcalculation.html` | 液压传动系统计算 |
| `/calculation/calculation_pneumaticdrivesystemcalculation.html` | 气压传动系统计算 |
| `/calculation/calculation_sealingsolutions.html` | 密封系统设计 |
| `/calculation/calculation_watersystemcalculation.html` | 水系统设计计算 |
| `/calculation/calculation_selectionofhydraulicbuffer.html` | 油压缓冲器选型 |
| `/calculation/calculation_rollingbearingdesigncalculation.html` | 滚动轴承设计与查询 |
| `/calculation/calculation_shaftdesign.html` | 轴设计计算 |
| `/calculation/calculation_electricalengineeringcalculation.html` | 电气工程计算 |
| `/calculation/calculation_motor(gearmotor)selection.html` | 电动机(减速机)选型计算 |
| `/calculation/calculation_tolerancefitquery.html` | 配合查询 |
| `/calculation/calculation_positiontolerance.html` | 位置公差查询 |
| `/calculation/calculation_boltconnection1.html` | 受横向载荷-铰制孔螺栓连接强度校核与设计 |
| `/calculation/calculation_boltconnection2.html` | 受横向载荷-紧螺栓连接强度校核与设计 |
| `/calculation/calculation_boltconnection3.html` | 受轴向载荷-紧螺栓连接（静载荷）校核与设计 |
| `/calculation/calculation_boltconnection4.html` | 受轴向载荷-紧螺栓连接（动载荷）校核与设计 |
| `/calculation/calculation_springs1.html` | 拉伸弹簧设计计算 |
| `/calculation/calculation_halfcirclekeyconnection.html` | 半圆键连接校核计算 |
| `/calculation/calculation_wedgekeyconnection.html` | 普通楔键连接校核计算 |
| `/calculation/calculation_tangentialkeyconnection.html` | 切向键连接校核计算 |
| `/calculation/calculation_rectangularsplineconnection.html` | 矩形花键连接校核计算 |
| `/calculation/calculation_involutesplineconnection.html` | 渐开线花键连接校核计算 |
| `/calculation/calculation_flatbelt.html` | 平带设计 |
| `/calculation/calculation_timingbeltdrive.html` | 同步带设计 |
| `/calculation/calculation_polyvbeltdesign.html` | 多楔带设计 |
| `/calculation/calculation_hydrauliccylindercalculation.html` | 液压油缸计算 |
| `/calculation/calculation_hydraulicpumpcalculation.html` | 液压泵计算 |
| `/calculation/calculation_hydraulicmotorcalculation.html` | 液压马达计算 |
| `/calculation/calculation_hydraulicjackcalculation.html` | 液压千斤顶计算 |
| `/calculation/calculation_oiltankthermalbalancecalculation.html` | 油箱热平衡计算 |
| `/calculation/calculation_pneumaticfingerclampingforcecalculation.html` | 气动手指夹紧力计算 |
| `/calculation/calculation_cylindergasconsumptioncalculation.html` | 气缸耗气量计算 |
| `/calculation/calculation_pneumaticcircuitcalculation.html` | 气动回路计算 |
| `/calculation/calculation_selectionofvacuumchuckandvacuumgenerator.html` | 真空吸盘与真空发生器 |
| `/calculation/calculation_chelionlinecalculationtool.html` | 气动供应商在线计算工具 |
| `/calculation/calculation_pressuredroponlinecalculation.html` | 压降在线计算 |
| `/calculation/calculation_deepgrooveballbearing.html` | 深沟球轴承设计与查询 |
| `/calculation/calculation_angularcontactballbearings.html` | 角接触球轴承设计与查询 |
| `/calculation/calculation_thrustballbearing.html` | 推力球轴承设计与查询 |
| `/calculation/calculation_taperedrollerbearing.html` | 圆锥滚子轴承设计与查询 |

## 外链：紧固件云（cn.fastencloud.com）小工具

| 工具 | 链接 |
|------|------|
| 千支重计算 | https://cn.fastencloud.com/tools/thousandWeight-gD.html |
| 铆钉通孔尺寸计算 | https://cn.fastencloud.com/tools/rivetThroughHoleSize-Ex.html |
| 铆钉长度计算 | https://cn.fastencloud.com/tools/rivetLength-oW.html |
| 铆钉的孔间距离计算 | https://cn.fastencloud.com/tools/distanceBetweenHolesRivet-ad.html |
| 最小破坏扭矩计算 | https://cn.fastencloud.com/tools/minBreakingTorque-rn.html |
| 螺纹极限尺寸计算 | https://cn.fastencloud.com/tools/limitSize-yP.html |
| 拧紧力矩计算 | https://cn.fastencloud.com/tools/tighteningTorque-Zn.html |
| 螺纹应力截面积计算 | https://cn.fastencloud.com/tools/whorlStress-kw.html |
| 毛坯直径计算 | https://cn.fastencloud.com/tools/roughDiameter-zM.html |
| 量规有效径计算 | https://cn.fastencloud.com/tools/gaugeEffectiveDiameter-3X.html |
| 螺栓和螺钉通孔计算 | https://cn.fastencloud.com/tools/boltScrewThroughHole-1Y.html |
| 米制螺纹镀层厚度和螺纹间隙计算 | https://cn.fastencloud.com/tools/platingThickness-8r.html |
| 米英制螺纹计算 | https://cn.fastencloud.com/tools/threadComparison-eR.html |
| 紧固件相关头型沉孔尺寸计算 | https://cn.fastencloud.com/tools/relevantHeadTypeCategory-5N.html |
| 三角螺纹填充率计算 | https://cn.fastencloud.com/tools/fillingRateTriangularThread-qQ.html |
| 螺栓突出螺母的末端长度计算 | https://cn.fastencloud.com/tools/endLengthDiameter-vA.html |
| 至和网 | https://www.zhwtk.com/ |
| CNC自学网 | https://www.cnczxw.com |
| 机械资源网 | https://www.u557.com/ |
| Baidu | https://www.baidu.com/ |
| Google | https://www.google.com |
| sogou | https://www.sogou.com/ |
| 360 | https://www.so.com/ |
| bing | https://cn.bing.com/ |
| Mechtool 百家号 | https://baijiahao.baidu.com/u?app_id=1685733001465637 |

## 已读两页要点

- **`/thread/index.html`：** 给出 ISO 与 GB/T 螺纹标准对照：ISO 68-1 ↔ GB/T 192-2003（基本牙型）、ISO 261 ↔ GB/T 193-2003（直径与螺距，一般用）、ISO 262 ↔ GB/T 9144-2003（选择系列）、ISO 724 ↔ GB/T 196-2003（基本尺寸）、ISO 965-1 ↔ GB/T 197-2003（公差）；另列 GB/T 2516-2003、GB/T 9145-2003、GB/T 3934-2003（页面原表）。
- **`/calculation/calculation_fastenercalculationtool.html`：** 只是把上表外链按名称汇总，不含站内计算。

## 阅读发现：来源、风险与站点自身的错误（子任务笔记汇总）

**来源与运营：** 全站未见版权声明、运营方、统一免责声明；联系方式只有「发邮件或微信公众号内私信我」。绝大多数公式和手册章节未标注出处；明示出处的有赫兹接触页（Johnson《Contact Mechanics》1985）、梁页（Gere & Goodno、Roark、Timoshenko、Hibbeler）、轴承页（GB/T 4662-2012、洛阳轴承研究所 1989 年图册、《全国滚动轴承产品样本》第 2 版）、橡胶术语（GB/T 5719-2006）。轴设计页材料表与材料重量页自定义密度表需登录。

**风险条目（页面原话，本库不引用）：** AutoCAD 页导航「序列号和密钥」「永久激活补丁」；MATLAB R2023b 页「内置补丁，你可以完全免费的进行使用」；ANSYS 2021 R2 页「附带了相对应的破解补丁…免费激活软件所有的功能」；SolidWorks 教程页以提取码分发 20 本原版培训教材（授权未写，疑似盗版，推测）；多个计算页提供国标与厂商样本的「pdf 在线浏览」，是否托管在本站看不出（推测有再分发风险）。

**子任务代入核算后确认的算错处：**
- 梁模拟器页：300×400 截面 I 写 1.6×10⁻⁵ m⁴（应为 1.6×10⁻³，差 100 倍）；100×200 截面 I 写 6.67×10⁻⁶（应为 6.67×10⁻⁵）。
- 梁可视化页：简支梁 L=4 m、w=10 kN/m、I=8×10⁻⁶ m⁴ 的最大挠度应约 20.8 mm，页面写 5.21 mm。
- 赫兹接触页：次表面剪应力深度一处 0.78a、示例 0.48a；「p max 与 F^(2/3) 成正比」与前文 F^(1/3) 矛盾。
- 平板临界载荷页：D=Et²/(12(1−ν²)) 的 t² 应为 t³。
- 分度凸轮改进等速页：c1=2π/(5π+4) 写「≈0.319112」，实际约 0.318813。
- 气动回路页算例「×30」实为 300。

**子任务指出的疑点（推测，未回教材 / 标准核对）：** 滚动轴承综合页力矩因数 fm 在试选与寿命两种模式方向相反，fn=(100/n)^(1/3) 与寿命式不自洽；轴设计页 ψ（0.3 与 0.6）疑似对调；滑动螺旋页锯齿形螺纹按 α=33° 取 cos(α/2) 疑似错误；凸轮分度器与液压缓冲器页单位未换算；矩形梁侧向稳定页 Iy 疑把强轴当弱轴；V 形刚架与单跨刚架页系数 N 不同。

**标准号自相矛盾：** 螺纹目录写 GB/T 197-2018，极限尺寸表注引 -2003；钢直尺页标题 GB/T 9056-2004、表题 -1988；技术制图术语页标题 GB/T 13361-2012、表题 -1992；挡圈页标题 -2017、注文 -1986；薄型平键标题 GB/T 1566、示例 1567；毡圈油封页并列 JB/ZQ 4606-1986（已废止）与 FZ/T 92010-1991；标准清单把 GJB 写成「国际军用标准」；GB/T 276 新页（-2013）自称数据仍按旧样本；大量新旧版并列。

**其他：** 抽取文本里有 OCR 式错字（弹簧表、轴承型号如「61084-2LS」「63/2-2Z」，可能是站点或抽取错误）；站内英汉词典称 14 万余条，来源未标注；教材、视频、样本网盘页可能涉及版权资源，本批文本无下载链接，需人工核查。

## 对 wiki 的映射

- [Mechtool 在线机械设计工具](../../wiki/entities/mechtool.md)
