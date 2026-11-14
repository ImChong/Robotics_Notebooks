# Spiderbot: An Open-Source Energy-Efficient Hexapod with Passive Gravity Compensation

- **作者：** Ritwik Sharma, Vimarsh Shah, Saransh Agrawal（Electronics and Robotics Club, BITS Pilani, KK Birla Goa Campus）
- **链接：** https://arxiv.org/abs/2609.26989 （项目主页 https://erc-bpgc.github.io/SpiderBot/ ）
- **状态：** primary_full_text_checked
- **核对范围：** 已通读 `arXiv:2609.26989` 论文全文（1–1021 行：摘要、四连杆被动重力补偿运动学与静力推导、Table I/II 物理参数与 BOM、Arduino Uno Q + MPU-6050 电子架构、mjlab 域随机化与非对称 Actor-Critic PPO 训练、Table VI–XII 平地/负重/坡面/台阶/崎岖地形与 Sim-to-Real 对比分析）
- **通读：** 已完成

## 核心要点提取

1. **四连杆被动重力补偿 12-DOF 机构（2-DOF/腿 + 弹簧）**
   - 相比常规 18-DOF（3-DOF/腿）六足机器人，Spiderbot 每条腿仅保留 **2 个主动自由度**（髋关节 `coxa` 与大腿 `femur`），用并行 **四连杆机构（Four-Bar Linkage）** 约束小腿 `tibia` 始终保持竖直，并嵌入 **2 根并联 450 N/m RC 减震弹簧**（总等效刚度 `900 N/m`，匹配三脚静立拐点 `880 N/m`）提供被动恢复力矩抵消重力。
   - **成本与静态能效**：整机 3D 打印 PLA 机身（重 ~2.59 kg，装电池整备 **3.0 kg**）搭配 off-the-shelf 元器件，总 BOM **不超过 400 美元**（Table II：6× ST3215 126 USD + 6× SC15 108 USD + Arduino Uno Q 44 USD + 12× 弹簧 18 USD + 2× Waveshare 舵机板 20 USD + 5200 mAh 2S LiPo 30 USD + PLA 10 USD + MPU-6050 IMU 2 USD）。
   - **静态站立功耗暴降**：站立功耗仅 **1.5 W**（比去弹簧版 15 W 降低 **>90%**，相比传统 3-DOF 平台如 PhantomX ~40 W、HexaV4 ~10 W 降低至 1/6–1/26，最高降幅 **96%**）；且挂载 **3.25 kg** 负载时静态站立功耗几乎不增（仅 **1.76 W** vs 去弹簧版 **33.92 W**；最大步态负重实测达 **3.75 kg**）。

2. **MuJoCo Tendon + mjlab 闭链仿真与无特权强化学习**
   - 标准 URDF/XACRO 无法描述并行四连杆闭链，论文利用 MuJoCo MJCF 等式约束与 `tendon` 弹簧建模，依托 GPU 并行的 `mjlab`（MJWarp + Isaac Lab manager API）在 **4096** 并行环境、`200 Hz` 物理步长 / `50 Hz` 控制频率（decimation=4）下用 **PPO**（3 层 MLP: `512-256-128`，非对称 Actor-Critic，5000 次迭代）跨 7 类程序化地形课程训练。
   - 对地面摩擦 `[0.3, 1.2]`、编码器偏置 `[-0.015, 0.015] rad`、CoM 偏移、PD 增益 `[0.8, 1.2]` 与关节转子惯量 `[0.25, 2.0]` 施加域随机化，Actor 仅使用本体感知，Critic 享受特权高度扫描信息。

3. **Sim-to-Real 与能耗-灵活性权衡实测（Table VI–XII）**
   - **平地（N=10）**：指令 `0.2 / 0.35 / 0.5 m/s` 下，仿真平均速度 `0.192 / 0.373 / 0.4305 m/s`（CoT `1.936 / 1.003 / 0.960`），实机平均速度 `0.163 / 0.186 / 0.231 m/s`（CoT `6.22 / 5.034 / 3.54`）。实机 CoT 约为仿真 3–4 倍，主因舵机驱动损耗、通信开销、反电动势及仿真未建模电气功耗。
   - **弹簧代价**：行走时弹簧引入约 **25%** 额外 CoT（如 `0.2 m/s` 下实机 CoT `6.22` vs 去弹簧 `4.97`），以换取 >90% 静态节能。
   - **非平地与工作空间边界**：实机与仿真均稳定攀爬 **5°/10°/15° 斜坡**、穿越自然土壤与不平地砖崎岖地形（实机 CoT `9.79` vs 仿真 `2.03`）；台阶越障在仿真可通过 **3.0 cm**（最大抬足高度 5 cm 的 60%），实机可通过 **2.0 cm**（40%），而在 **5.0 cm**（100%）两者均失败——证实 2-DOF 四连杆固定小腿竖直约束带来硬性几何越障上限。

## 局限性与边界条件

- 固定 `900 N/m` 弹簧刚度针对 3.25 kg 静态零额外力矩点设计，无法随任务负载在线变刚度调节。
- 2-DOF 机构小腿始终保持竖直且缩减径向工作空间，无法跨越超过抬足上限的垂直台阶或压低身姿钻行。
- 串行舵机（ST3215/SC15）仅作位置环响应，仿真机械功率与实机总电功率存在 >3× CoT 差距，高动态冲击能力有限。
