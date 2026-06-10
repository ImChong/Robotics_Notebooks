# 电机 TN/TI 曲线与电磁仿真软件 FAQ（维护者整理）

- **类型**：`personal`（对话/答疑整理，非正式出版物）
- **日期**：2026-06-10
- **原始对话**（仅作溯源，不在 wiki 建独立节点）：
  - 电机 TN 曲线解析：<https://chatgpt.com/share/6a28dcd5-e694-83ea-a209-75b2bb666de3>
  - 电机分析软件推荐：<https://chatgpt.com/share/6a28dcef-bbd0-83ea-ae58-db0ae99bfbef>
- **用途**：为 [电机转矩-转速曲线（TN 曲线）](../../wiki/concepts/motor-torque-speed-curve.md)、[电机转矩-电流曲线（TI 曲线）](../../wiki/concepts/motor-torque-current-curve.md)、[电机电磁仿真软件选型](../../wiki/comparisons/motor-em-simulation-software.md) 提供可追溯编译来源；正文以 wiki 页为准。

## 对话 1 要点：TN 曲线与 TI 曲线

### TN 曲线（Torque-Speed Curve）

- 横轴转速 \(N\)（rpm），纵轴转矩 \(T\)（Nm）；描述不同转速下可输出力矩。
- **恒转矩区**：低速段电流达上限，转矩近似恒定，功率 \(P = T\omega\) 随转速上升。
- **恒功率区**：超过基速后受母线电压限制，进入弱磁，功率近似恒定、转矩随转速反比下降。
- 人形/腿足选型关注：**峰值转矩**（起跳、抗冲击）、**基速**（摆腿与最高关节速度）、**连续转矩**（行走站立的真实工作区；厂家宣传的峰值往往只能维持数秒）。
- 测试报告常配套：效率地图、电流-转速、功率-转速；TN 是评估关节模组的第一张图。

### 功率换算

- 机械功率：\(P = T\omega\)；转速用 rpm 时常写 \(P_{\mathrm{kW}} = T \cdot n / 9550\)。
- 高力矩低速未必高功率；跑步、快速摆腿等动态动作更看 **峰值功率** 而非峰值转矩 alone。

### TI 曲线（Torque-Current Curve）

- 横轴电流 \(I\)（A），纵轴转矩 \(T\)（Nm）；PMSM/BLDC 理想关系 \(T = K_t I_q\)。
- 斜率即力矩常数 \(K_t\)（Nm/A）：同样转矩所需电流越小，铜损 \(I^2R\) 越低、发热压力越小。
- 高电流区曲线变平可能来自磁饱和、温升、驱动限流或 MTPA 失效。
- 从连续/峰值电流与 \(K_t\) 可反推连续/峰值转矩，比单独标称峰值更有参考价值。

## 对话 2 要点：电机分析软件

三类典型仿真与常用工具：

| 物理域 | 典型输出 | 常用软件 |
|--------|----------|----------|
| 电磁 FEA | 磁密、齿槽转矩、反电势、Ld/Lq、TN | Maxwell、JMAG、Motor-CAD、Flux |
| 流体 CFD | 风道、搅风损耗、散热 | Fluent、Motor-CAD 散热模块 |
| 热分析 | 绕组/磁钢温升、连续功率 | Motor-CAD、Icepak、JMAG 热 |

行业常见组合：

- **机器人关节厂**：Maxwell（电磁）+ Motor-CAD（热/效率地图）。
- **汽车电驱**：JMAG + Motor-CAD。
- **高校多物理场**：COMSOL 一套耦合电磁-热-流体。

人形关节电机学习顺序建议：Motor-CAD（最快出 TN/效率/温升）→ Maxwell（行业标准）→ JMAG（高端 IPM/伺服）。
