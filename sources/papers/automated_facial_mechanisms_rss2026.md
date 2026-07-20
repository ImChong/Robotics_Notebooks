# Automated Synthesis of Facial Mechanisms for Conversational Animatronic Robots

> 来源归档

- **标题：** Automated Synthesis of Facial Mechanisms for Conversational Animatronic Robots
- **类型：** paper
- **出处：** 2026 · RSS 2026 Finalist · arXiv preprint
- **arXiv：** <https://arxiv.org/abs/2607.11688>
- **论文 HTML：** <https://arxiv.org/html/2607.11688>
- **项目页：** <https://zzongzheng0918.github.io/automated-facial-mechanisms-synthesis/>
- **代码：** <https://github.com/ZZongzheng0918/automated-facial-mechanisms-synthesis>（**已开源**）
- **作者：** Zongzheng Zhang, Zi Lin; 通讯 Hang Zhao（清华 AIR / BAAI / IIIS / BUAA 联合）
- **入库日期：** 2026-07-20
- **一句话说明：** 参数化连杆驱动面部模板 + 分层自动合成流水线，针对任意面部几何自动生成可制造 Animatronic 机构方案；RSS 2026 Finalist，代码已公开。

---

## 核心摘录（策展，非全文）

### 问题与动机

- 会话型 Animatronic 机器人需要能表达表情的面部机构，但现有设计流程完全依赖人工经验，无法快速适配新面部几何。
- 核心挑战：(i) 连杆机构参数与面部几何的高维耦合；(ii) 可制造性约束（舵机力矩、结构干涉）；(iii) 多 AU 协同覆盖。

### 关键贡献

1. **参数化面部机构模板：** 将面部驱动抽象为连杆子机构集合，每个子机构对应一组 FACS AU；模板通过少量参数跨越多样化面部几何。
2. **分层自动合成流水线：** 形状适配层 → 运动学优化层 → 可制造性修整层，全程无需人工介入。
3. **端到端制造验证：** 多款 3D 打印 + 硅胶皮肤原型，真机表情驱动演示。

### 方法要点

| 维度 | 描述 |
|------|------|
| 输入 | 目标面部 3D 几何 + 期望 AU 集合 |
| 模板 | 参数化连杆骨架，可形变适配目标曲面 |
| 第一层 | 形状适配：将模板铰接点映射到目标面部表面 |
| 第二层 | 运动学优化：最大化 AU 覆盖，最小化干涉风险 |
| 第三层 | 可制造性：舵机力矩/行程/安装约束收敛 |
| 输出 | CAD 文件 + 舵机参数表 + 驱动映射 |

### 实验摘要

- 在多款不同面部几何（人物肖像、角色造型）上验证，输出可制造方案并实际装配驱动。
- 表情序列（说话、微笑、皱眉等）可通过生成机构复现，视觉效果自然。

### 局限（论文自述范围）

- 皮肤非线性弹性未完整建模；复杂联合 AU 可能引入干涉；依赖高质量面部几何输入。

### 对 wiki 的映射

- [paper-automated-facial-mechanisms-animatronic](../../wiki/entities/paper-automated-facial-mechanisms-animatronic.md)
- [manipulation](../../wiki/tasks/manipulation.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2607.11688>
- 项目页：<https://zzongzheng0918.github.io/automated-facial-mechanisms-synthesis/>
- 代码：<https://github.com/ZZongzheng0918/automated-facial-mechanisms-synthesis>
