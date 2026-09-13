# wechat_zanehub_joint_module_self_development_workflow

> 来源归档（blog / 微信公众号）

- **标题：** 自研机器人关节模组，需要走过什么流程？
- **类型：** blog
- **作者：** Zane Zhang（公众号署名）
- **原始链接：** https://mp.weixin.qq.com/s/sN92pyUm26oRHasfkbyxpg
- **入库日期：** 2026-09-13
- **一句话说明：** 从需求定义、传动构型、五件套选型、结构设计、标定控制、装配制造到四层测试矩阵，串起自研旋转关节模组的完整工程链路；强调负载谱而非单点峰值扭矩、扭矩密度与批次一致性。
- **沉淀到 wiki：** [`wiki/concepts/joint-module-self-development-workflow.md`](../../wiki/concepts/joint-module-self-development-workflow.md)

## 核心摘录（归纳，非全文）

### 1) 需求定义：三个问题与指标瀑布

- **立项三问**：载荷谱、空间约束、量产成本——未对齐则构型/选型/测试全线返工。
- **负载谱优先**：持续 RMS 扭矩定发热与寿命，峰值扭矩定启停/冲击余量（常乘 1.5–2 安全系数），转速曲线定高速段，循环次数定减速器/轴承等级。
- **扭矩密度**：ρ = T额定 ÷ m（N·m/kg）；工业机器人约 50–120，人形追求更高；峰值扭矩常按 2–3 倍连续预留。
- **量级参考**：0.8 m 双足（~12 kg）上肢 ~6 N·m、下肢 ~36 N·m；1.7 m 级（~60 kg）上肢 ~60 N·m、下肢 ~200 N·m。
- **指标瀑布**：整机行为 → 关节峰值扭矩/额定扭矩/转速/行程/背隙/刚度/重量/外径/中空内径/电压平台。

### 2) 传动构型

| 构型 | 特点 | 典型位置 |
|------|------|----------|
| 谐波 | 近零背隙、高减速比、轻小；柔轮怕持续冲击 | 肩、肘、腕 |
| 行星 | 刚性、抗冲击、效率高；背隙较大 | 腿足启停冲击 |
| RV | 高刚度大负载 | 髋/基座 |
| 准直驱 | 低减速比、反驱好、响应快 | 腿足动态控制 |

### 3) 五件套选型要点

- **无框力矩电机**：定转子入壳；内转子响应快、外转子扭矩大；连续瓶颈在散热。
- **双编码器**：电机端 + 输出端（19 bit 双绝对值已普及）；磁编 EMC 与抱闸/穿线干扰需隔磁。
- **抱闸**：制动力矩常取额定力矩 1.3–1.5 倍；摩擦片式为主。
- **力矩传感器（可选）**：应变/磁弹；标定（零点、串扰、温漂）是难点。
- **驱动器**：FOC；EtherCAT / CANopen；24–48 V；协议须在立项锁死。

### 4) 结构 / 标定 / 制造 / 测试

- **结构**：同轴度与气隙、刚度链（谐振频率）、中空走线（>20 mm 级）、密封 IP54/67、仿真（静强度/疲劳/模态/热磁耦合/EMC）。
- **标定与控制**：零点对齐、传动误差补偿、三环 + 力矩环、总线时序多轴同步。
- **装配**：定子压装、轴承预紧、柔轮啮合、润滑定量——一致性靠自动化产线全检追溯。
- **测试四层**：性能（TN/背隙/刚度/温升/噪声）、耐久（谐波 8000–10000 h 级、急停数千次）、环境（-20~55 ℃/振动/EMC）、整机联调。
- **量产**：DFMEA/PFMEA、MTBF 目标、降额设计；批次离散度决定整机调参与售后成本。

## 对 wiki 的映射

- [joint-module-self-development-workflow](../../wiki/concepts/joint-module-self-development-workflow.md)（本次升格主页面）
- [motor-design-workflow](../../wiki/overview/motor-design-workflow.md)（电机子链路；本页覆盖减速器/传感/装配/验收）
- [humanoid-hardware-101-integrated-actuators](../../wiki/overview/humanoid-hardware-101-integrated-actuators.md)（集成执行器语境）
- [humanoid-knee-harmonic-drive-limits](../../wiki/concepts/humanoid-knee-harmonic-drive-limits.md)（谐波 vs 行星/RV 分工）
- [motor-torque-speed-curve](../../wiki/concepts/motor-torque-speed-curve.md)（TN 读图与测试对账）

## 可信度与使用边界

- 第三方工程归纳，扭矩量级与寿命标称为行业通行口径，具体数值以目标机型台架与供应商 datasheet 为准；无单一厂商项目页需核查开源状态。

## 当前提炼状态

- [x] 文章基础摘要填写
- [x] 初步 wiki 页面映射确认
