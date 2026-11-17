# Closing the Loop：在 MuJoCo 中还原 Unitree G1 推杆踝闭链

- **类型：** 官方作者技术博客 / 原始工程记录（非 arXiv 论文）
- **英文标题：** Closing the loop — How to model the Unitree G1's push-rod ankle in MuJoCo
- **中文标题：** 闭合运动学回路：在 MuJoCo 中建模宇树 G1 推杆踝关节
- **作者：** Kevin Zakka、Qiayuan Liao
- **发布日期：** 2026-11-16
- **原文：** <https://kzakka.com/closing-the-loop/>
- **模型下载：** <https://kzakka.com/closing-the-loop/assets/g1_ankle_models.zip>
- **关联技术：** MuJoCo、Unitree G1、闭链运动学、空间腱、固定腱、反射惯量、Sim2Real
- **入库日期：** 2026-11-17
- **核对范围：** 已逐节通读作者原文（机械限位实测、固定腱限位、闭链运动学与两种实现、力矩/惯量映射、LAFAN1 评测、重置求解、积分器稳定性与模型选型、下载说明、参考文献及许可）。
- **通读：** 已完成

## 原始资料身份与开放状态

- 这是作者的**博客技术文章**，不应登记为具有 arXiv ID 的独立论文。
- 文章附有五份 G1 踝模型及 mesh 的 ZIP：串联踝一份，闭链空间腱/实体推杆各有电机侧、踝侧驱动版本；要求 **MuJoCo ≥ 3.16**，作者实验使用 **4.0.1**。
- 文章直接展示 MJCF 片段与 Python 闭环初始化函数 `equality_jacobian`、`assemble`、`close_crank`，但页面**未提供独立的项目源码 GitHub 仓库、训练脚本或可核验的 CLI**。
- **可获取状态：** 文章与 ZIP 下载链接公开；本次未成功下载并逐文件审查 ZIP 内 XML/mesh，因此模型包中的配置细节、独立执行入口与实际运行结果**尚未本地验证**。不得把原文代码片段写成已运行的脚本。
- **许可：** 原文文字与图 CC BY 4.0、代码和模型 Apache 2.0；G1 mesh 为 Unitree 设计，**不在上述两项许可范围内**，应遵守 ZIP 中 NOTICE。
- **对应 wiki：** [G1 推杆踝闭链 MuJoCo 建模](../../wiki/entities/g1-pushrod-ankle-closed-loop-modeling.md)。

## 原文关键论点与实测（仅代表作者报告）

1. **实际机械限位并非 pitch–roll 矩形。** 中性 pitch 附近，roll 手动实测约 ±16–18°；在 pitch 极值附近允许 roll 明显收窄。作者保留 ±15° roll 约束，再用四个线性不等式裁去矩形四角，以 MuJoCo fixed tendon 上限实现。拟合边界与个别轨迹仍有最高约 2° 的偏差。
2. **闭链传动随姿态变化。** 两根推杆对曲柄位置与踝部锚点施加固定距离约束；令驱动角 `u=h(x)`、足部 `x=[pitch,roll]`，则 `u_dot=G x_dot`；虚功给出 `tau_x=G^T tau_u`，电机侧惯量反射到踝端为 `I_x=G^T I_u G`。这不是两只独立串联踝驱动能表达的常量关系。
3. **闭链实现有层次。** 空间腱 + tendon equality 只表达理想无质量杆的几何/力传递；如需推杆质量、惯量、碰撞或端部关节，则用实体推杆、两摆动铰链与 `connect` equality（实体杆模型额外约束与自由度较多）。
4. **LAFAN1 实验与边界。** 采用旧矩形约束的重定向中，1.5% 的支撑脚姿态比拟合边界超出 1°；在拟合止挡内且能求解电机角的承重脚样本中，少于 0.1% 需要某电机超过 25 N·m（逆动力学与近似接触力估计，**非真机电机实测**）。
5. **控制器失配。** 标称姿态中，按电机分别施加 14.25 N·m/rad 并不等价于训练用踝端 28.5 N·m/rad：作者计算 pitch 约 62、roll 约 19 N·m/rad。在一项 6 s 舞蹈、256 对配对 rollout 中，独立电机增益故障率 20%，用 `G^{-T}` 映射踝端控制力矩的结果为 16%。这是单项策略仿真实验，不是广义成功率保证。
6. **初始化与积分器。** 闭链重置要同时设置踝角和电机角/速度，满足 rod-length equality；文中给出 Gauss–Newton 解和 G1 曲柄解析解。对踝侧位置执行器，作者在 4 ms 下观察到 `implicitfast` 闭环误差发散；`discrete` 在其测试步长范围内稳定。对电机侧驱动，原文测试的积分器均稳定。

## 工程选型结论

- 重定向或初期 RL：可以先用**串联踝 + 实测拟合止挡**，避免策略请求物理不可达角度。
- 涉及电机侧力矩/速度/惯量/PD 增益：选**空间腱闭链**进行部署前验证。
- 涉及推杆自重、碰撞、端部摩擦/限位：考虑**实体推杆闭链**，但计算与约束成本更高。
- 无论哪种闭链：需做**约束一致重置**，并验证实际执行器位置与积分器组合的数值稳定性。
- 文中模型尚未逐文件复核；下载后应独立检查 XML 的 `qpos0`、equality、actuator、mesh 许可及指定积分器。

## 来源与交叉

- 作者原文与交互图：<https://kzakka.com/closing-the-loop/>
- 作者的 MuJoCo 模型 ZIP：<https://kzakka.com/closing-the-loop/assets/g1_ankle_models.zip>
- [Wiki：G1 推杆踝闭链建模](../../wiki/entities/g1-pushrod-ankle-closed-loop-modeling.md)
- [Wiki：人形机器人并联关节解算](../../wiki/concepts/humanoid-parallel-joint-kinematics.md)
