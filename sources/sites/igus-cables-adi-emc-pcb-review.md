# igus 运动电缆与 ADI/TI 电磁兼容资料复核

- **类型：** 官方技术资料 / 原文复核
- **核对日期：** 2026-11-16
- **核对范围：** igus Flexible Cables for Industrial Motion Applications 完整正文、产品分类、应用例和 FAQ；ADI MT-095 Rev.0（01/2009）16 页全文文字、公式、表与图注；TI ISO7741 产品页完整 Product details、Features、Description 技术正文。图以 PDF 提取的文字与图注核对，未做电磁仿真或实物 EMC 认证。MT-031 虽已下载但未通读，不列为本记录已读原文。
- **通读：** 已完成

## 原文与版本

- [igus motion cables](https://www.igus.com/cables)；本次读取材料 SHA-256：`95737cdf78790f7c70910847d8420da4c38643425c233a0c04c1416ff08929db`。
- [ADI MT-095](https://www.analog.com/media/en/training-seminars/tutorials/MT-095.pdf)；本次读取材料 SHA-256：`72279b86679ea1c7e111284b1f009a4fd24efb866ce4bf7c3e5a67597633eea1`。
- [TI ISO7741](https://www.ti.com/product/ISO7741)；本次读取材料 SHA-256：`2df4774ffd54d4f5852e969ed60fe9dc5b1acf71f54cf7646ef5a84018987d65`。

## 核对结论

- igus 区分电机、反馈、总线、混合和机器人扭转电缆；材料耐油/UV/温度与寿命适用性按具体电缆类型确定。厂家最大循环次数或保修宣传不能变成所有关节的寿命保证。
- MT-095 按源、路径、接收器分析干扰：共阻抗传导、容性/感性近场耦合与远场辐射是不同路径；降低回路面积、双绞、屏蔽与滤波分别针对不同耦合机制。
- 导电屏蔽适合电场和高频磁场；低频磁场通常需要高磁导率材料。开口的最长尺寸、接缝、连接器与穿壳线缆会决定真实屏蔽效果。
- 屏蔽接地需按干扰频率、电缆电长度及安全要求设计；高频常需两端低阻抗/360°连接，低频地环问题可用适合电路的混合接地处理，不能宣称两端接地总是错误。
- ISO7741 为三正向一反向的数字信号隔离器，不集成隔离电源；应同时安排隔离侧供电。无 F 与 F 后缀的默认输出不同，耐压与爬电间距也随封装变化。

## 对 wiki 的映射

- [humanoid-wire-harness-types-protection](../../wiki/concepts/humanoid-wire-harness-types-protection.md)
- [joint-module-electromagnetic-isolation](../../wiki/concepts/joint-module-electromagnetic-isolation.md)
