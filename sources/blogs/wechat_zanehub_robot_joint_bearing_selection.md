# 机器人关节用轴承选型经验分享

> 来源归档（blog / 微信公众号）

- **标题：** 机器人关节用轴承选型经验分享
- **类型：** blog
- **作者：** Zane Hub（Zane Zhang；第三方工程解读，非厂商官方）
- **原始链接：** https://mp.weixin.qq.com/s/rweTJtjvt8LaJLLM8eEYBg
- **入库日期：** 2026-09-17
- **抓取方式：** HTTP 正文提取（WebFetch；本环境无 Camoufox 快照）
- **原始抓取落盘：** [`sources/raw/wechat_zanehub_robot_joint_bearing_selection.md`](../raw/wechat_zanehub_robot_joint_bearing_selection.md)
- **一句话说明：** 按载荷路径梳理谐波/RV/直驱关节轴承布置；交叉滚子、柔性轴承、薄壁与四点接触选型；五笔账（载荷/刚度/精度/寿命谱/游隙预紧）+ 润滑温升 + 装配坑与失效判据。
- **沉淀到 wiki：** [`wiki/concepts/robot-joint-bearing-selection.md`](../../wiki/concepts/robot-joint-bearing-selection.md)
- **姊妹文：** [`wechat_zanehub_joint_module_self_development_workflow.md`](wechat_zanehub_joint_module_self_development_workflow.md)、[`wechat_zanehub_humanoid_leg_knee_why_not_harmonic.md`](wechat_zanehub_humanoid_leg_knee_why_not_harmonic.md)、[`wechat_zanehub_humanoid_mass_production_experience.md`](wechat_zanehub_humanoid_mass_production_experience.md)

## 核心摘录

### 1) 先看载荷路径，再翻样本

六轴/人形关节载荷为 **径向 + 轴向 + 倾覆力矩** 复合；不同传动布置轴承位置完全不同：

| 关节类型 | 典型轴承 |
|----------|----------|
| **谐波关节** | 电机轴小轴承；波发生器内 **柔性轴承**；输出端 **交叉滚子**（常集成进减速器） |
| **RV 关节** | 主轴承承全部外载；内部摆线轮/偏心轴/太阳轮支承轴承（单台 RV 约 **9–15** 套） |
| **直驱关节** | 大直径 **四点接触球** 或 **交叉滚子** 直接支承转子 |

### 2) 交叉滚子轴承（输出端主力）

- 一套承受双向轴向、径向与倾覆力矩；滚子 90° 交错 + 隔离块。
- **RB 型**（外圈分半）关节输出常用；精度 **P5**，关键轴 **P4**；设计寿命目标行业惯例 **≥6000 h**。
- 当量动载荷含倾覆项：\(P = X F_r + Y F_a + Z M\)；静安全系数 \(S_0 = C_0/P_0\)，滚子一般 **≥1.5**。

### 3) 谐波柔性轴承

- 套圈极薄，随椭圆凸轮周期性变形；**柔轮–外圈过盈** + 专用压装工装。
- 疲劳寿命常为整机短板；参照 **GB/T 30819-2024**（2025-04-01 实施）；几乎全定制，备件须同型号同批次。

### 4) 薄壁 / 四点接触

- 腹肘腕空间紧张处；四点接触可替代角接触配对。
- **薄 = 对安装面圆度/平面度/过盈极敏感**；须同步写形位公差与工艺。

### 5) 五笔选型账

1. **载荷谱**：用实测/仿真谱 + Miner 累积，勿单峰值放大一档。
2. **刚度**：减速器刚度与轴承刚度串联；预紧提刚度但增摩擦与温升。
3. **精度等级**：P5/P4 常见；回差大头在齿隙与装配——做 **装配链误差预算**。
4. **寿命**：ISO 281 / GB/T 6391，滚子 \(\varepsilon=10/3\)；99% 可靠度乘 \(a_1\)；温升 **+10℃ ≈ 脂寿命减半**。
5. **游隙/配合**：交叉滚子常负游隙预紧；过盈会吃掉 **60–70%** 径向游隙；座轴同轴度建议 **≤0.02 mm**。

### 6) 润滑、温升与现场判据

- 交叉滚子脂润滑为主（ISO VG 68+）；填充量 **25–35%**；正常温升宜 **≤30℃**（RV 类常按 30℃ 考核）。
- 停机四信号：温升趋势上行、周期咯噔/高频嘘声、振动增大、回差渐进增大。

**对 wiki 的映射**

- [robot-joint-bearing-selection](../../wiki/concepts/robot-joint-bearing-selection.md)
- 交叉 [Hardware 101 · 直线与轴承](../../wiki/overview/humanoid-hardware-101-linear-transmission-bearings.md)、[自研关节模组流程](../../wiki/concepts/joint-module-self-development-workflow.md)
