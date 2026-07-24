# 电机 / 关节测功机（一手资料索引）

> 来源归档（ingest）

- **标题：** 测功机原理、选型与机器人关节台架验收：标准、厂商手册与开源实现一手摘录
- **类型：** standard / manual / repo / site（合集）
- **入库日期：** 2026-07-24
- **一句话说明：** 汇总 GB/T 43200、IEC 60034-2-1、Magtrol 磁滞/涡流测功机手册、开源四象限对拖台架与国内关节对拖测试方案，作为「样机台架如何测出 TN/TI」的原始依据。
- **沉淀到 wiki：** 是 → [`wiki/concepts/motor-dynamometer.md`](../../wiki/concepts/motor-dynamometer.md)

## 为什么值得保留

- [电机设计流程](../../wiki/overview/motor-design-workflow.md) 与 [力矩电机设计纵深 Stage 6](../../roadmap/depth-torque-motor-design.md) 都要求 **对拖/测功台架** 出整机 TN/TI、效率地图与温升，但此前仓库缺少测功机本身的一手资料。
- 「测功机」不是单一设备：磁滞吸收、涡流吸收、磁粉制动、电力对拖（四象限）各有零速扭矩、连续功率与回馈能力边界；选型错了会测不出堵转或烧毁制动器。
- 人形关节还要把 **电机单体台架** 与 **一体化关节模组台架**（减速器 + 编码器 + 总线力矩模式）分开验收——国标与厂商方案条目互证。

## 核心摘录

### 1) GB/T 43200-2023 — 机器人一体化关节性能及试验方法

- **来源：** 国家市场监督管理总局 / 国家标准委；标准信息服务：
  - [全国标准信息公共服务平台 · GB/T 43200-2023](https://openstd.samr.gov.cn/bzgk/std/newGbInfo?hcno=B2E40B3445ACE9E166E8E402E89853AF)
  - [国家标准全文公开系统检索页](https://std.samr.gov.cn/gb/search/gbDetailed?id=053404E3EE3F8F91E06397BE0A0A9209)
- **元数据：** 发布 2023-09-07，实施 2024-04-01；英文名 *Performance and related test methods of mechatronic joints for robots*；CCS J28 / ICS 25.040.30；归口 SAC/TC 591。
- **要点：**
  - 规定 **机器人一体化关节** 的性能与试验方法；适用于协作机器人与腿足式机器人关节，其他类型可参照。
  - 验收对象是 **机电一体化关节**（电机 + 传动 + 传感/驱动集成），不是裸电机 datasheet 单点。
  - 国内关节模组测功/对拖方案普遍 **对标本标准** 的机械 / 电气 / 控制三类试验项（见 AIP 条目）。
- **分册归档：** [`gbt_43200_2023_robot_joint_performance.md`](gbt_43200_2023_robot_joint_performance.md)
- **对 wiki 的映射：** [motor-dynamometer](../../wiki/concepts/motor-dynamometer.md)

### 2) IEC 60034-2-1:2024 — 旋转电机损耗与效率试验方法

- **来源：** [IEC Webstore — IEC 60034-2-1:2024](https://webstore.iec.ch/en/publication/67756)（第 3 版，取代 2014 第 2 版）
- **要点：**
  - 定义旋转电机 **损耗与效率** 的标准试验方法（不含牵引车辆电机）。
  - **直接法**：电功率输入 + 机械输出（扭矩 × 转速）→ 效率；扭矩测量装置通常为测功机或联轴扭矩传感器，精度等级有要求。
  - **间接法 / 损耗分离**：无可靠机械测功条件时用分项损耗求和；不确定性通常高于直接法。
  - 亦定义 **双机对拖（back-to-back）**：两台同型机机械耦合，由电功率差求总损耗——与机器人实验室「电力测功 / 对拖」架构同构。
- **分册归档：** [`iec_60034_2_1_motor_efficiency.md`](iec_60034_2_1_motor_efficiency.md)
- **对 wiki 的映射：** [motor-dynamometer](../../wiki/concepts/motor-dynamometer.md)、[motor-torque-speed-curve](../../wiki/concepts/motor-torque-speed-curve.md)

### 3) Magtrol — 磁滞 / 涡流 / 磁粉测功机手册与 M-TEST

- **来源：**
  - 手册索引：[Magtrol Manuals](https://www.magtrol.com/manuals/)
  - HD/ED 磁滞测功机用户手册：[`hdmanual.pdf`](https://www.magtrol.com/wp-content/uploads/hdmanual.pdf)（2023-10）
  - WB32 高速涡流测功机：[`wb32manual.pdf`](https://www.magtrol.com/wp-content/uploads/wb32manual.pdf)
  - M-TEST 7 软件说明：[`mtest7.pdf`](https://www.magtrol.com/wp-content/uploads/mtest7.pdf)
- **要点：**
  - Magtrol 将吸收式制动分为三类：**Hysteresis（HD）**、**Eddy-current（WB）**、**Powder（PB）**；选型先看最大扭矩，再看 **最大机械功率（散热）**，最后看最高转速。
  - **磁滞制动**：扭矩与转速近似无关，可从空载斜坡到 **堵转（locked rotor）**；摩擦less 磁耦合；连续功率受冷却（对流 / 压缩空气 / 风机）限制，间歇额定常按 ≤5 min 另给曲线。
  - **涡流制动**：扭矩随转速上升，额定扭矩在额定转速附近达到；适合高速；水冷定子可耗散更高连续功率。
  - **磁粉制动**：零速即可出额定扭矩，偏低速大扭矩。
  - 机械功率：\(P[\mathrm{W}] = T[\mathrm{N\cdot m}] \cdot n[\mathrm{min^{-1}}] \cdot 1.047\times10^{-1}\)（手册 SI 式）。
  - M-TEST 7：Ramp / Curve / Manual / Pass-Fail / Coast / Overload-to-trip；Curve 模式用于扫速–扭矩–电流–输入/输出功率，是工业侧出 **TN 曲线** 的软件入口。
- **分册归档：** [`magtrol_dynamometer_manuals.md`](magtrol_dynamometer_manuals.md)
- **对 wiki 的映射：** [motor-dynamometer](../../wiki/concepts/motor-dynamometer.md)

### 4) Capo01 — ODrive 开源四象限电力测功机

- **来源：** [Capo01/odrive_based_electric_motor_dynamometer](https://github.com/Capo01/odrive_based_electric_motor_dynamometer)
- **要点：**
  - **电力对拖**：被测电机经联轴器接吸收电机（D6374 级）；吸收机可制动也可驱动 → **四象限**（电动/发电）。
  - 扭矩由 **负载传感器 + 摆臂** 测反应力矩；母线电流分流估算电功率；ODrive + Python 脚本出数据文件。
  - 能力量级：约 50–2000 W、0–7500 rpm、峰值制动 ~3.5 N·m；可测效率地图、静态 \(K_t\)、相电阻/电感（室温与升温）、空载最高转速。
  - 制动能量经母线闭环回馈被测侧，降低电源容量——与工业再生测功同思路的实验室低成本实现。
- **开源状态：** **已开源**（硬件照片、流程与样例数据在仓内；项目 README 标注曾暂停维护）。
- **分册归档：** [`../repos/odrive_based_electric_motor_dynamometer.md`](../repos/odrive_based_electric_motor_dynamometer.md)
- **对 wiki 的映射：** [motor-dynamometer](../../wiki/concepts/motor-dynamometer.md)

### 5) AIP 艾普 — 机器人旋转关节 / 人形电机对拖测试方案

- **来源：**
  - [机器人旋转关节模组测试系统](https://aipuo.com/products/1336.html)
  - [人形机器人电机测试系统](https://www.aipuo.com/products/1171.html)
  - [旋转关节测试方案案例（对标 GB/T 43200）](http://www.aipuo.com/index.php/case/1341.html)
- **要点（厂商一手产品说明，非学术论文）：**
  - 台架形态：**双轴对拖**（被测关节 + 负载测功机）+ 高精度扭矩/转速传感器 + 功率分析仪 + 同步采集。
  - 试验项三分：
    - **机械**：反向启动转矩、背隙、刚度、惯量、制动转矩等；
    - **电气**：T-N、最高/额定转速、瞬时最大转矩、转矩常数、密度；
    - **控制**：定位/转矩精度、阶跃响应、位置/转速/转矩频带、波动系数、温升、振动噪声。
  - 明确宣称对标 **GB/T 43200-2023**；人形大关节与空心杯小关节分系统覆盖。
- **开源状态：** **确认未开源**（商业测控设备；无公开固件/机械图纸仓库）。
- **分册归档：** [`aip_robot_joint_dynamometer.md`](aip_robot_joint_dynamometer.md)
- **对 wiki 的映射：** [motor-dynamometer](../../wiki/concepts/motor-dynamometer.md)

### 6) 艾诺（Ainuo）— 人形电机「三层测试」工程分层（厂商技术文）

- **来源：** [人形机器人电机测试，这几种检测手段缺一不可](https://www.yiqi.com/technews/detail_7c6b4054aa115176.html)（青岛艾诺仪器，2026-07-01，仪器网转载公开资料）
- **要点：**
  - **灵巧手空心杯**：低电感 → 匝间脉冲上升沿 / 百 MHz 采样与毫欧级电阻；
  - **关节电机**：齿槽与摩擦扭矩（1–30 rpm、mNm 级 + FFT 阶次）、堵转/低启电流；
  - **关节模组**：旋变/霍尔相位、CAN/EtherCAT 连通与整模组同步采集——已超出「纯测功机」进入系统验收。
- **定位：** 工程分层叙事，补齐「测功机只覆盖动态扭矩层」的边界；产品型号为厂商宣传，wiki 只取分层逻辑。
- **对 wiki 的映射：** [motor-dynamometer](../../wiki/concepts/motor-dynamometer.md)

## 推荐继续阅读（外部）

- Magtrol [DSP7010 Dynamometer Controller](https://www.magtrol.com/manuals/) — 闭环扭矩/转速控制与 PC 接口
- IEEE 112 / CSA 390 — 北美电机效率试验方法（与 IEC 60034-2-1 间接法对照时用）

## 当前提炼状态

- [x] 标准 + 厂商手册 + 开源对拖 + 国内关节对拖方案摘录与 wiki 映射
- [x] 沉淀概念页 `wiki/concepts/motor-dynamometer.md`
- [ ] 后续可补：具体扭矩传感器精度等级选型表、台架安全互锁 checklist
