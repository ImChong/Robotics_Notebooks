# FlyBrainLab

> 来源归档

- **标题：** FlyBrainLab
- **类型：** repo
- **机构：** Fruit Fly Brain Observatory / Columbia 等
- **链接：** https://github.com/FlyBrainLab/FlyBrainLab
- **论文：** Lazar et al., *Accelerating with FlyBrainLab the Discovery of the Functional Logic of the Drosophila Brain in the Connectomic Era*, eLife 2021 — [10.7554/eLife.62362](https://dx.doi.org/10.7554/eLife.62362)
- **Docker：** https://hub.docker.com/r/fruitflybrain/fbl
- **Stars：** ~58+（2026-09）
- **入库日期：** 2026-09-10
- **一句话说明：** 从果蝇连接组数据构建可执行神经回路的交互计算平台，整合 3D 探索、回路编译与 Neurokernel GPU 仿真。
- **代码：** https://github.com/FlyBrainLab/FlyBrainLab（**已开源**）
- **沉淀到 wiki：** 是 → [`wiki/entities/flybrainlab.md`](../../wiki/entities/flybrainlab.md)

---

## 三大能力（README / eLife）

1. **3D 探索与可视化** — 果蝇脑数据浏览（NeuroNLP / BrainMapsViz）
2. **可执行回路构建** — 从探索数据直接编译仿真回路
3. **交互式功能逻辑探索** — 计算研究者配置回路、实验科学家交互验证

---

## 安装模式

| 模式 | 说明 |
|------|------|
| 仅用户端 | `pip install flybrainlab[full]`，默认连公共后端（无 GPU 回路执行） |
| 完整安装 | 用户端 + 本地 Neurokernel 后端，需 NVIDIA GPU |
| Docker | `fruitflybrain/fbl` 镜像 |
| AMI | `ami-02218ae5a3d1fd06d`（AWS GPU 试用） |

```bash
conda create -n flybrainlab python=3.9 git -y
conda activate flybrainlab
python -m pip install flybrainlab[full] neuromynerva
```

---

## 与连接组数据关系

- 可对接 FlyWire / neuPrint 等来源构建 **executable circuits**
- 与 Male CNS 的 **结构—功能** 研究互补：Male CNS 提供雄性全 CNS 布线；FlyBrainLab 侧重 **仿真验证**

---

## 开源核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| 用户端 | **已开源** |
| Neurokernel 后端 | 完整 GPU 仿真需本地/Docker/AMI 部署 |
| 公共后端 | 仅可视化，**不含** GPU 回路执行 |

---

## 对 wiki 的映射

- [FlyBrainLab](../../wiki/entities/flybrainlab.md)
- [果蝇连接组工具栈](../../wiki/concepts/fly-connectomics-stack.md)
- [FlyWire](../../wiki/entities/flywire.md)
