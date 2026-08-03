# 动力学仿真驱动的人形机器人下肢衍生式设计（华中科技大学学报 2026）

> 来源归档（ingest）

- **标题：** 动力学仿真驱动的人形机器人下肢衍生式设计 / Generative design of humanoid robot leg driven by dynamic simulation
- **类型：** paper / humanoid / hardware design / generative design / electro-hydraulic hybrid / dynamics simulation
- **DOI：** [10.13245/j.hust.260645](https://doi.org/10.13245/j.hust.260645)
- **期刊页：** <http://xb.hust.edu.cn/thesisDetails#10.13245/j.hust.260645&lang=zh>
- **CNKI（境外 DOI 跳转）：** <https://link.oversea.cnki.net/doi/10.13245/j.hust.260645>
- **期刊：** 华中科技大学学报（自然科学版），2026，54(6): 1–7；栏目 Intelligent Robotics / 智能机器人
- **收稿 / 纸质出版：** 2025-10-27 / 2026-06-23
- **作者：** 罗元春、纵怀志、周蕾*（通信）、张军辉
- **机构：** 浙江大学流体动力基础件与机电系统全国重点实验室；中国航空工业集团公司西安飞行自动控制研究所（AVIC FACRI）
- **基金：** 国家自然科学基金杰出青年基金（52525505）；人形机器人学科突破先导项目（JYB2025XDXM208）；博士后创新人才支持计划（BX20250423）；控制执行机构技术创新中心开放基金（ICCA618-202406）
- **代码 / 项目页：** **无**（期刊论文；站点未列 GitHub / 数据集；`oa=0`）
- **入库日期：** 2026-08-03
- **一句话说明：** 用跳跃工况动力学仿真提取多工况载荷，再以衍生式设计（非传统拓扑去料）生成电液混合驱动人形下肢大/小腿连杆；大腿/小腿减重 62.5%/61.6%，跳跃高度 0.303→0.327 m。

## 开源状态（核查，2026-08-03）

- **确认未开源：** 学报详情页与 PDF 均未给出代码、CAD 或数据集链接；附件类型仅有网络 PDF / WORD / Meta-XML。
- **全文获取：** 学报页「阅读全文 PDF / PDF 下载」→ `getAttachFileId(attachCode=lowqualitypdf)` → OSS 签名 PDF（约 3.3 MB，7 页）。
- **边界：** 可复现的是**方法流程与量化结果口径**，不是可下载的 Fusion/Simscape 工程。

## 摘要级要点

- **问题：** 经验削薄 / 材料替换 /「先设计后校核」与动态载荷弱耦合；传统拓扑优化依赖固定初始几何，难系统考察多工况可制造拓扑。
- **构型：** 单腿 **5 DoF**（髋滚转+俯仰、膝俯仰、踝俯仰+滚转）；大/小腿长 **400 mm**；骨盆静载 **30 kg**；原地竖跳目标 **0.3 m**。
- **驱动：** **电液混合**——髋/膝俯仰用自研 **EHA**（3.18 kg，推力 12 kN，速度 540 mm/s）；髋滚转与踝用电机直驱（云深处 J80-27P / J60-10）。髋 EHA 三角连杆、膝 EHA 四连杆避死点；踝双电机+平行四边形解耦。
- **仿真：** Simscape Multibody + PD；起跳质心离地初速 2.43 m/s；落地瞬间膝俯仰力矩峰值 **2188.9 N·m**；WEBOTS 交叉验证起跳关键关节力矩误差 **<3%**；基线跳跃高度 **0.303 m**。
- **衍生式设计：** Fusion 360；材料 Ti6Al4V + 金属增材；四工况（落地冲击 / 起跳驱动 / 意外冲击 / 失衡）；保留体+障碍体+可生长空间；选 **y 向增材约束**方案。
- **结果：** 大腿 7.75→2.90 kg（−62.5%）、小腿 5.97→2.29 kg（−61.6%）；FEA 等效应力 592.7 / 643.5 MPa；一阶模态 156.83 / 70.25 Hz；重仿真跳跃 **0.327 m**。

## 核心论文摘录（MVP）

### 1) 电液混合 + 连杆传动布置

- **链接：** §1.2–1.3；图 1–2
- **摘录要点：** 大力矩关节（髋/膝俯仰）走 EHA+连杆，姿态/带宽敏感关节走电机直驱；膝用四连杆规避伸直/深蹲死点。
- **对 wiki 的映射：**
  - [动力学仿真驱动的人形下肢衍生式设计](../../wiki/entities/paper-humanoid-leg-generative-design-dynamics.md)
  - [人形整机机械布局设计](../../wiki/concepts/humanoid-mechanical-layout-design.md)
  - [人形腿部行星滚柱丝杠直线驱动](../../wiki/concepts/planetary-roller-screw-humanoid-leg-actuation.md) — 对照另一类直线推力路线

### 2) 仿真提取工况 → 衍生式多工况优化

- **链接：** §2–3.2；表 1；图 6
- **摘录要点：** 起跳匀加速定执行器选型，落地冲击定结构强度；衍生式不以预设几何去料，而在保留/障碍约束下生长。
- **对 wiki 的映射：**
  - [动力学仿真驱动的人形下肢衍生式设计](../../wiki/entities/paper-humanoid-leg-generative-design-dynamics.md)
  - [Actuator 102 · 负载与质量螺旋](../../wiki/overview/humanoid-actuator-102-load-and-mass-spiral.md)

### 3) 减重与动态性能闭环验证

- **链接：** §3.3–4；图 7–8
- **摘录要点：** 强度/疲劳/模态/瞬态通过后，仅改质量重仿真，跳跃高度再抬约 8%。
- **对 wiki 的映射：**
  - [动力学仿真驱动的人形下肢衍生式设计](../../wiki/entities/paper-humanoid-leg-generative-design-dynamics.md)
  - [Humanoid Hardware 101 · 机身与材料](../../wiki/overview/humanoid-hardware-101-chassis-materials.md)
  - [人形整机硬件设计纵深路线](../../roadmap/depth-humanoid-hardware-design.md)

## BibTeX

```bibtex
@article{luo2026generative,
  title   = {动力学仿真驱动的人形机器人下肢衍生式设计},
  author  = {罗元春 and 纵怀志 and 周蕾 and 张军辉},
  journal = {华中科技大学学报(自然科学版)},
  year    = {2026},
  volume  = {54},
  number  = {6},
  pages   = {1--7},
  doi     = {10.13245/j.hust.260645}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-humanoid-leg-generative-design-dynamics.md`](../../wiki/entities/paper-humanoid-leg-generative-design-dynamics.md)
- 互链：[人形整机机械布局设计](../../wiki/concepts/humanoid-mechanical-layout-design.md)、[机身与材料](../../wiki/overview/humanoid-hardware-101-chassis-materials.md)、[负载与质量螺旋](../../wiki/overview/humanoid-actuator-102-load-and-mass-spiral.md)、[硬件选型](../../wiki/queries/humanoid-hardware-selection.md)、[整机硬件设计路线](../../roadmap/depth-humanoid-hardware-design.md)
