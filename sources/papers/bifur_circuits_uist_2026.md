# Bifur-circuits: Interactive and Modular Metamaterial Building Blocks Via Bifurcated Geometries

> 来源归档（ingest · 2026-09-17）

- **标题：** Bifur-circuits: Interactive and Modular Metamaterial Building Blocks Via Bifurcated Geometries
- **类型：** paper
- **作者：** Marwa AlAlawi, Regina Zheng, Abdullah Negm, Jiaji Li, Emma Li, Yasuaki Kakehi, Yoshihiro Kawahara, Junyi Zhu, Ticha Sethapakdi*, Stefanie Mueller*
- **机构：** 麻省理工（MIT）；东京大学；密歇根大学
- **会议：** UIST 2026
- **项目页：** <https://hcie.csail.mit.edu/research/Bifur-circuit/bifur-circuits.html>
- **视频：** <https://youtu.be/eCUYbfXCvME>
- **入库日期：** 2026-09-17
- **一句话说明：** 将 auxetic kirigami 单元与 AB/C 连接器做成机械+电气双模块化超材料积木，bifurcation 扩展构型空间，内嵌 I2C 感知构型，支持多材料 3D 打印与交互设计工具链。

## 核心摘录（策展，非全文）

1. **动机：** 传统 modular metamaterial 多限于 2D/2.5D、手工多步制造，或机械/电气层分离；单 auxetic 单元通常仅 3 个稳态，限制可表达几何。
2. **Bifur-circuits 构造：** auxetic kirigami **metamaterial unit**（三稳态：开/闭/闭）+ **Type AB**（正交）与 **Type C**（共线）**connector**；T-joint 卡扣 + 5° 锥度保证单一插入方向；导电 Filaflex 走线于非导电 TPU/PLA 体内。
3. **构型空间：** 单单元 3 态；perpendicular bifurcation 两单元可达 5+ 构型；$N(m,n,p) \propto 3^{m \times n \times p}$（理想上界）；2×2×1 闭环示例 22+ 种区分构型（含 connector 朝向）。
4. **感知原理：** 单元内缘接触激活状态专属电路；connector 四端口 I2C（SDA/SCL + R1/R2 状态线 + J1/J2 朝向线）；root 递归扫描发布 JSON 拓扑图。
5. **制造：** 单元 multimaterial FDM（Sainsmart TPU 95A + Filaflex 92A；双稳/三稳插 N45 磁铁）；connector 为 PLA 刚性体 + 导电 trace；PCB 热压贴合。
6. **评测：** I2C 发现延迟 $t_{detection} = 256.2 m - 472$ ms（$R^2=0.9727$，20 单元链）；10k 次同向/反向/交替压缩后电阻与铰链完整性保持。
7. **讨论局限：** 导电 TPU 电阻率导致 Vcc/GND 跨块供电不可靠（每块独立 LiPo ~45 min 连续感知）；检测延迟随单元数线性增；TPU 黏弹性带来 12h+ 构型记忆偏置。
8. **机器人相关应用（MIT News）：** 可重构夹爪、康复辅助具、灾后可变形 shelter、可重构天线（团队前序 Meta-antenna 延伸）。

## 对 wiki 的映射

- [paper-bifur-circuits](../../wiki/entities/paper-bifur-circuits.md)
- 项目页：[bifur-circuits.md](../sites/bifur-circuits.md)

## 参考来源（原始）

- 项目页：<https://hcie.csail.mit.edu/research/Bifur-circuit/bifur-circuits.html>
- MIT News：<https://news.mit.edu/2026/mit-engineers-create-system-for-building-shape-changing-smart-devices-0827>
- 视频：<https://youtu.be/eCUYbfXCvME>
