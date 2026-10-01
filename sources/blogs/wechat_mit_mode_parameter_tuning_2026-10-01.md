# MIT模式参数整定的完整方法

> 来源归档（blog / 微信公众号）

- **标题：** MIT模式参数整定的完整方法
- **类型：** blog
- **作者：** 待核实（WebFetch 未解析公众号 nick_name；正文为 6 自由度清洁机械臂 MIT 阻抗整定实操）
- **原始链接：** https://mp.weixin.qq.com/s/ocrp_k3INGYsI3GzzeUZJQ
- **入库日期：** 2026-10-01
- **抓取方式：** WebFetch 直拉 `mp.weixin.qq.com` 正文
- **原始抓取落盘：** [`sources/raw/wechat_mit_mode_parameter_tuning_2026-10-01.md`](../raw/wechat_mit_mode_parameter_tuning_2026-10-01.md)
- **一句话说明：** 关节 MIT 阻抗模式 $\tau=K_p e_q + K_d \dot e_q + \tau_{ff}$ 的参数全景与六关节差异表；纯手感实验、Pinocchio RNEA 半模型、全 RNEA 前馈三条整定路径；J5→J3 优先序、阶跃验收、分阶段 Kp/Kd 与 J5/J6 重力耦合对策；含验收清单与安全限幅建议。
- **步骤 2.5（开源核查）：** 工程方法论 + Pinocchio 引用，**无**单一官方项目页；Pinocchio 为 [已开源](../../sources/repos/pinocchio.md) 第三方库。
- **沉淀到 wiki：** [`wiki/methods/mit-joint-impedance-mode-tuning.md`](../../wiki/methods/mit-joint-impedance-mode-tuning.md)

## 核心摘录

### 控制律与参数

$$\tau = K_p (q_{des}-q) + K_d (\dot q_{des}-\dot q) + \tau_{ff}$$

| 参数 | 含义 | 整定 |
|------|------|------|
| $K_p$ | 虚拟弹簧刚度 | 实验：弹簧感但不振荡 |
| $K_d$ | 虚拟阻尼 | 常取 $(0.05\sim0.2) K_p$，拍击无来回摆 |
| $\tau_{ff}$ | 重力/摩擦/惯性前馈 | 竖直/俯仰关节静态悬浮标定；摩擦匀速正反扫 |

### 六关节差异（文内量级）

- **J1/J2 竖直旋转：** 高 $K_p$（200–500 Nm/rad 量级），$\tau_{ff}$ 以摩擦为主。
- **J3 升降：** 单位 N/mm；**重力补偿关键**。
- **J5/J6 俯仰：** 重力矩大；J5+J6 平行轴 **重力耦合** 需解耦补偿。
- **J4 轻载：** $K_p$ 可低。

### 三条路径

- **A 纯实验：** $K_d=0,\tau_{ff}=0$ 找 $K_p$ → 加 $K_d$ → 重力/摩擦标定（非控制背景可操作）。
- **B 半模型（推荐）：** Pinocchio `rnea(q,0,0)` 作重力 $\tau_{ff}$；摩擦仍实验；$K_p$ 可略降。
- **C 全模型：** 实时 $M(q)$ RNEA 全前馈；低速清洁任务收益有限。

### 验收（节选）

- 空载阶跃：无超调、无振荡、稳态 $<0.05°$。
- J3/J5/J6 松手不下滑（仅靠 $\tau_{ff}$）。
- 连续 30 min 电机温升 $<60°C$；力矩超额定 80% 自动降 $K_p$。

## 对 wiki 的映射

- 方法页：[mit-joint-impedance-mode-tuning](../../wiki/methods/mit-joint-impedance-mode-tuning.md)（新建）
- 关联：[阻抗控制](../../wiki/concepts/impedance-control.md)、[重力补偿](../../wiki/concepts/gravity-compensation.md)、[MIT 紧凑帧总线语义](../../wiki/overview/motor-drive-firmware-bus-protocols.md)、[Pinocchio](../../wiki/entities/pinocchio.md)
