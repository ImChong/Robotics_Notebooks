# 无框力矩电机设计需要的知识和工具链有哪些？ / Frameless Torque Motor Design: Skills and Toolchain

- **类型：** blog / 微信公众号
- **作者：** Zane Hub（第 404 篇原创）
- **发布时间：** 2026-11-28 08:00（原文标注）
- **原始链接：** [微信公众号原文](https://mp.weixin.qq.com/s/sJa2_3W_mMU0FdarfHxO3Q) · [本次读取入口](https://api.readgzh.site/rd?s=sJa2_3W_mMU0FdarfHxO3Q)
- **入库日期：** 2026-11-28
- **核对范围：** 通过用户所给读取入口通读作者正文开篇及十节（架构、电磁、热、结构、退磁、反馈与驱动、标准与试验、工具、失效、流程）；基于官方安装指南、软件手册、应用报告与标准目录核对关键接口和口径。
- **通读：** 已完成
- **对 wiki 的映射：** [关节电机设计流程：无框力矩电机设计与工具链](../../wiki/overview/motor-design-workflow.md)
- **源码：** 原文是设计与工具选型指南，未发布可执行配套代码；不适用源码运行时序图。

## 要点归纳

1. 无框套件把安装后气隙、转子支撑和散热交给模组设计方。把负载谱、机械包络、母线、电流上限和冷却边界放在同一张工况表。
2. 从尺寸量级和热阻网络推到电磁、热、承载变形与控制仿真。关键模型输出是可测量的反电势、损耗、温升、气隙与低速转矩波动，不是软件截图。
3. 试验需区分断电齿槽、通电转矩纹波和供电限幅。用安装态实测数据更新模型，再发布连续及限时峰值的转矩–转速包络。

## 核验与使用边界

| 核验点 | 结论 |
|---|---|
| 运行气隙与热边界 | Kollmorgen 的安装指南要求用户提供足够刚性的轴承、达到机型图纸同心度与工作间隙要求；其安装建议不替代具体规格。 |
| 仿真耦合 | Motor-CAD 手册给出电磁损耗进热模型、温度反向影响电磁性能并迭代求解的模式，也支持集总热网络的稳态与瞬态计算。 |
| 参数和计算口径 | 用平均切向应力定义时，圆柱气隙转矩为 `2πr²Lσ`，不能把未经统一定义的电负荷与磁密直接代入经验系数。统一相/线、峰值/RMS、机械/电气角速度；更稳妥的电机常数表述是 `T/√Pcu`，与 `Kt/√R` 的具体换算取决于电流、电阻及三相铜损口径。 |
| 控制边界与示例 | TI 应用报告给出理想平衡三相线性 SVPWM 的相电压有效值上限 `Vdc/√6`（相峰值 `Vdc/√3`）；它不是实机反电势容许值，还要留电阻/电感压降和调节裕量。原文 §2.6 的示例按其所写纹波近似式代数应为 `48/(6×200e-6×16000)=2.5 A`；这是量级估算，并非通用 PWM 纹波模型。 |
| 动平衡与验收 | ISO 官方目录标示 ISO 1940-1:2003 已撤销；刚性转子的程序与公差入口为 ISO 21940-11（含修订）。本文不将作者举例的平衡等级、耐压电压、材料牌号或偏差阈值当作通用验收条件。 |

## 一手资料

- [Kollmorgen 无框电机安装指南](https://www.kollmorgen.com/sites/default/files/Framless%20motor%20Mounting%20%26%20Installation%20Guidlines.pdf)
- [Ansys Motor-CAD 电磁与热耦合](https://ansyshelp.ansys.com/public/Views/Secured/MotorCAD/v252/en/Motor-CAD_UG/MotorCAD/topics/electromagnetic__thermal_model.html) · [集总热网络](https://ansyshelp.ansys.com/public/Views/Secured/MotorCAD/v252/en/Motor-CAD_UG/MotorCAD/topics/thermalmodellingofelectricmotors.html)
- [Texas Instruments SPRA524：空间矢量 PWM](https://www.ti.com/lit/an/spra524/spra524.pdf)
- [ISO 21940-11:2016 官方目录](https://www.iso.org/standard/54074.html)（仅核对标准范围及替代关系）
