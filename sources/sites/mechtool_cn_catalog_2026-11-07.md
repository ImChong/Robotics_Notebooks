# Mechtool 在线机械设计工具（站点目录归档）

> 来源归档（site / 在线机械设计资料与计算工具站）

- **标题：** Mechtool 在线机械设计工具
- **类型：** site
- **原始链接：** https://www.mechtool.cn/
- **入库日期：** 2026-11-07
- **抓取方式：** curl 取首页 HTML，并对站内全部 190 个链接逐个请求（返回 HTTP 200 的 186 个，已记录页面标题与描述）；另读了 `/thread/index.html`（公制、美制和英制螺纹）与 `/calculation/calculation_fastenercalculationtool.html`（紧固件计算工具）两页全文。
- **核对范围：** 首页、全站链接目录、上述两页读完；**其余 184 页只记录了标题 / 描述 / 状态码，没有逐页读正文，也没有运行任何在线计算器、没有核对其公式**；站点未标注各页数据所依据的手册版本，本库不把其中任何数值当作已核对的一手资料。
- **通读：** 未完成
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

## 对 wiki 的映射

- [Mechtool 在线机械设计工具](../../wiki/entities/mechtool.md)
