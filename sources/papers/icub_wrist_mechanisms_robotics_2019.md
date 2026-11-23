# iCub 人形机器人腕部机构比较（Shah et al., Robotics 2019）

> 来源归档（paper · 原始论文全文通读）

- **英文标题：** A Comparison of Robot Wrist Implementations for the iCub Humanoid
- **中文标题：** iCub 人形机器人腕部机构实现方案比较
- **类型：** paper
- **作者：** Divya Shah, Yuanqing Wu, Alessandro Scalzo, Giorgio Metta, Alberto Parmiggiani
- **机构：** 意大利技术研究院（Istituto Italiano di Tecnologia, iCub Facility）、热那亚大学（University of Genoa）、博洛尼亚大学（University of Bologna）
- **期刊：** *Robotics* 2019, 8(1), 11（2019-02-17）
- **DOI：** [10.3390/robotics8010011](https://doi.org/10.3390/robotics8010011)
- **官方 HTML：** <https://www.mdpi.com/2218-6581/8/1/11>
- **官方 PDF：** <https://www.mdpi.com/2218-6581/8/1/11/pdf>
- **版本：** 期刊正式发表版（2019）
- **入库日期：** 2026-11-23
- **核对范围：** 出版社正式 HTML 全文的第 1–8 节（引言、执行器重定位、iCub mk.2、设计目标、六种 CAD 模型、工作空间、讨论与结论），以及图 1–8、表 1–2、参考文献与作者贡献。
- **通读：** 已完成
- **代码状态：** 论文作者贡献部分提到用于 Quaternion Joint 运动学建模的求解器，但正文和出版页面未给出对应公开代码仓库、可下载的 CAD 模型或复现脚本；**未核实到可运行的官方实现**（核查日 2026-11-23）。

## 研究问题与设计边界

iCub mk.2 手-前臂组件共有 12 个独立自由度，重量约 0.95 kg。本文专注其中 **2-DOF 腕部俯仰 / 偏航**；roll 由前臂内的上游机构提供，不在本次比较范围。原始腕部采用远置的 Faulhaber 直流电机、159:1 行星齿轮与绳传动，降低腕端质量但引入传动弹性与耦合。

作者期望的新腕具备：接近半球的姿态工作空间（理想倾角至 90°）、低奇异性、低输入耦合、高各向同性、紧凑尺寸及低运动质量。可用腕机构体积近似为上/下端直径 70/50 mm、高 150 mm 的截锥体。

## 核心论文摘录与核对

1. **统一比较对象（§4–5）：** 串联 Gimbal 与现有 iCub mk.2 为两项基准；候选并联机构为球面五杆、球面六杆、OmniWrist-III（4-UU）与 Quaternion Joint，共六种。原摘要将研究概括为若干两自由度姿态机构，读比较表时应按六列方案计数。
2. **CAD 与工作空间方法（§5–6）：** 使用 PTC Creo Parametric 4.0 / Mechanism 为结构建模，对可行电机输入角组合按网格采样，求末端平台位置与姿态并做归一化工作空间比较；对 Quaternion 关节的近似球面滚动约束，使用误差最小化的逆运动学处理 CAD 容差问题。
3. **基准 iCub mk.2（§3、表 1）：** 两轴机械可动范围分别为 \([-56°,56°]\) 和 \([-38°,38°]\)。作者给出两电机角度到关节角的耦合关系：\(\theta_1=1.38q_1\)，\(\theta_2=1.38q_1+1.38q_2\)。
4. **姿态范围（§6–7、表 2）：** Gimbal、OmniWrist 与 Quaternion 具备完整半球工作空间；球面五杆的 pitch ±90° / yaw ±30°，球面六杆的 pitch ±90° / yaw ±45°。
5. **控制上的差异（§6–7、表 2）：** Gimbal 完全解耦；iCub mk.2 与球面五/六杆部分解耦；OmniWrist、Quaternion 不完全解耦。OmniWrist 的电机角到平台位置映射有不对称 warping，Quaternion 相对更规整。
6. **不可忽略的取舍（表 2）：** Quaternion Joint 虽被认为最有潜力，但在表 2 的「平台到基座距离恒定」一项为 No，而其他五种为 Yes；不能误记为一切指标最优。
7. **结论与证据级别（§8）：** 作者建议继续推导 Quaternion 的解析运动学并优化参数，**不是** 已完成替换 iCub 腕部或做过真机控制对照试验。论文为 CAD 模型工作空间研究。

## 关键对比（原文 Table 2）

| 机构 | 驱动解耦 | ROM / 工作空间 | Warping | 平台距离恒定 |
|---|---|---|---|---|
| 2-DOF Gimbal | 完全 | ±90°；完整半球 | 否 | 是 |
| iCub mk.2 | 部分 | Pitch ±56°；Yaw ±38°；部分半球 | 否 | 是 |
| 球面五杆 | 部分 | Pitch ±90°；Yaw ±30°；部分半球 | 否 | 是 |
| 球面六杆 | 部分 | Pitch ±90°；Yaw ±45°；部分半球 | 否 | 是 |
| OmniWrist-III | 无 | Tilt 90°；完整半球 | **是** | 是 |
| Quaternion Joint | 无 | Tilt 90°；完整半球 | 否 | **否** |

> 角度与对比标签取自论文表 2。倾角 Tilt 与 Gimbal 的独立 pitch/yaw 坐标不应直接视为同一参数化。

## 对 wiki 的映射

- **独立论文详情：** [iCub 腕部机构比较](../../wiki/entities/paper-icub-wrist-mechanisms-comparison.md)
- **概念交叉引用：** [人形机器人并联关节解算](../../wiki/concepts/humanoid-parallel-joint-kinematics.md)
- **纵深路线：** [人形整机硬件设计 / Stage 1](../../roadmap/depth-humanoid-hardware-design.md)

## 使用注意

- 没有公开代码或可核对的复现实验入口，因此不应虚构运行时序图或声称 CAD 结果经过真机验证。
- 原文的“Quaternion 方案较适合”是针对 iCub 特定空间、传动及控制约束下的选型判断，并非通用腕关节优劣排序。
