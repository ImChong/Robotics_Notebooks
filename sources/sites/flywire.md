# FlyWire

> 来源归档

- **标题：** FlyWire
- **类型：** site
- **机构：** FlyWire Consortium（Princeton / MRC LMB / Allen / Janelia 等）
- **链接：** https://flywire.ai/
- **Codex：** https://codex.flywire.ai/
- **入库日期：** 2026-09-10
- **一句话说明：** 首个经大规模专家 proofreading 的成年果蝇全脑连接组平台（雌性），含 ~140K 神经元、50M+ 突触与社区注释。
- **代码：** 探索前端与 API 由 Consortium 维护；注释更新见 [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations)
- **沉淀到 wiki：** 是 → [`wiki/entities/flywire.md`](../../wiki/entities/flywire.md)

---

## 规模快照（官网，2024-10 Nature 旗舰论文后）

| 指标 | 数值 |
|------|------|
| Proofread 神经元 | **139,255**（中枢脑 + 视叶） |
| 突触 | **50M+**（含神经递质信息） |
| 社区注释 | **100K+** cell labels |

---

## 旗舰论文（Nature 2024）

- Dorkenwald et al., *Neuronal wiring diagram of an adult brain*
- Schlegel et al., *Whole-brain annotation and multi-connectome cell typing*
- Matsliah, Yu et al., *Neuronal "parts list" and wiring diagram for a visual system*

---

## 技术栈要点

- **成像：** Janelia Bock lab 高分辨率 EM；Bock / Saalfeld lab 对齐
- **自动重建：** Princeton Murthy / Seung lab；平台供 Consortium proofreading
- **3D 查看器：** Google Research 开发（Neuroglancer 系）
- **突触预测：** Janelia Funke / Saalfeld lab；神经递质：Funke + Jefferis lab
- **探索入口：** **Codex**（Connectome Data Explorer）

---

## 与 Male CNS 关系

- FlyWire 为 **雌性** 全脑连接组；Male CNS 为 **雄性** 全 CNS（脑 + VNC）。
- Male CNS 项目页的 **Dimorphism Explorer** 与 Neuroglancer 场景内 **共注册雌性样本 mesh**，支持跨性别回路比较。

---

## 开源核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| 连接组数据 | **开放**（Codex 探索 + 程序化工具） |
| 注释仓库 | **部分开源**（`flywire_annotations` GitHub） |
| 平台源码 | 非单一公开 monorepo；依赖 Neuroglancer 等组件 |

---

## 对 wiki 的映射

- [FlyWire](../../wiki/entities/flywire.md)
- [Male CNS Connectome](../../wiki/entities/male-cns-connectome.md)
- [果蝇连接组工具栈](../../wiki/concepts/fly-connectomics-stack.md)
