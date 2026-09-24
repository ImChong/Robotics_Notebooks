# 机器人力控关节模组常用的力矩传感器类型和主流品牌有哪些？

> 来源归档（blog / 微信公众号）

- **标题：** 机器人力控关节模组常用的力矩传感器类型和主流品牌有哪些？
- **类型：** blog
- **作者：** Zane Hub（Zane Zhang；第三方工程解读，非厂商官方）
- **原始链接：** https://mp.weixin.qq.com/s/_Rs6EmAOlqxzP_rLTppWqQ
- **入库日期：** 2026-09-24
- **抓取方式：** WebFetch（curl 遇 CAPTCHA；Camoufox 本环境未预装）
- **原始抓取落盘：** [`sources/raw/wechat_zanehub_joint_torque_sensor_types_2026-09-24.md`](../raw/wechat_zanehub_joint_torque_sensor_types_2026-09-24.md)
- **一句话说明：** 力控关节力矩感知四路线（电流环估力、双编码器差值、应变片物理传感、SAW/磁弹性非接触）+ 关节级 vs 六维力边界 + 国际/国内品牌与模组厂商选配趋势 + 七参数选型清单。
- **沉淀到 wiki：** [`wiki/concepts/joint-torque-sensor-selection.md`](../../wiki/concepts/joint-torque-sensor-selection.md)
- **姊妹文：** [`wechat_zanehub_joint_module_self_development_workflow.md`](wechat_zanehub_joint_module_self_development_workflow.md)、[`wechat_zane_zhang_joint_encoder_comparison_2026-09-23.md`](wechat_zane_zhang_joint_encoder_comparison_2026-09-23.md)、[`wechat_zanehub_robot_joint_bearing_selection.md`](wechat_zanehub_robot_joint_bearing_selection.md)

## 核心摘录

### 1) 电流环估力矩

- $\tau_m \approx K_t \cdot I_q$；关节端 $\tau_j \approx \eta \cdot i \cdot K_t \cdot I_q$
- 库仑/黏性摩擦随温度、转速慢漂移；谐波柔轮变形、齿隙、偏心使映射非线性且时变
- **够用：** 碰撞检测（撞没撞）；**不够：** 恒力打磨、精密装配

### 2) 双编码器差值（变相力矩估计）

- $\Delta\theta = \theta_{\mathrm{out}} - \theta_m / i$；$\hat\tau_j \approx K_{\mathrm{eq}} \cdot \Delta\theta$
- 17 位编码器谐波单关节最小可检测力矩约 **0.5 N·m** 量级
- 测的是传动链**累计变形**（外力 + 内部摩擦 + 弹性混在一起），无法解耦
- **零差云控 eRob T 版** 即此路线；零增厚、零刚度损失、零额外传感器成本

### 3) 物理传感器（应变片主流）

- 串在减速器之后、输出法兰之前；人形旋转执行器成本约 **30%**（次于谐波 ~36%，高于无框电机 ~13.5%）
- 惠斯通全桥，灵敏度 2–3 mV/V；标定后精度 **0.1–0.5% FS**
- 代表：**Bota BTS-T**（±25–300 N·m，7 mm 薄盘，4 kHz，500% 过载）、**宇立 M221X**、**坤维 KWR61N150**、**FUTEK**、**Aidin ATSB**
- **SAW**（Sensor Technology TorqSense）：无源转子、非接触；关节大批量场景尚少
- **磁弹性**：大型台架为主，关节内少见

### 4) 关节力矩 vs 六维力

- **关节力矩传感器：** 单轴、嵌入关节、参与实时控制环
- **六维 F/T：** 末端法兰、测交互力螺旋；协作臂常 **关节级 + 末端六维** 分层；人形以关节级为主，腕/踝等交互密集处加六维

### 5) 选型七参数

1. 量程与过载（如 500% 过载）
2. 刚度代价：$1/K_{\mathrm{joint}} = 1/K_{\mathrm{gear}} + 1/K_{\mathrm{sensor}} + 1/K_{\mathrm{shaft}}$
3. 精度、迟滞、重复性（全温区综合误差）
4. 带宽与延迟（高端 ~4 kHz 采样、EtherCAT）
5. 温漂与温补
6. 结构接口（中孔、外径、7 mm 薄盘 + 谐波法兰）
7. 标定与数据闭环（能否进驱动器控制环）

### 6) 四点判断（文内结论）

1. 分层感知：关节单轴 + 末端六维，不可互替
2. 应变片在可见周期内仍是主流
3. 双编码器差值是「够用哲学」，非权宜
4. 人形放量推动该供应链国产化与降本

## 对 wiki 的映射

- 升格 [`joint-torque-sensor-selection`](../../wiki/concepts/joint-torque-sensor-selection.md) 概念页
- 交叉 [`joint-encoder-selection`](../../wiki/concepts/joint-encoder-selection.md)、[`joint-module-self-development-workflow`](../../wiki/concepts/joint-module-self-development-workflow.md)、[`actuator-drive-chain-selection-loop`](../../wiki/queries/actuator-drive-chain-selection-loop.md)
