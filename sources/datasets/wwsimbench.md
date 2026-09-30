# WWSimBench / WWBench 仿真资产（Hugging Face）

> 来源归档

- **标题：** WWSimBench（WuWen benchmark · WWBench 开放资产）
- **类型：** dataset（仿真 3D 资产包，USD 为主）
- **链接：** <https://huggingface.co/datasets/Wuwen-AI/WWSimBench>
- **发布方：** 无问芯穹（Wuwen-AI / Infinigence AI）
- **版本：** **0.1**（README）
- **入库日期：** 2026-09-30
- **开源状态：** **已发布** — 数据集在 Hugging Face 公开下载；**无** 独立 GitHub 评测运行器或任务 JSON 仓（截至入库日仅资产 + README）。
- **一句话说明：** 面向 **桌面操纵、铰链开合、线缆插拔、长程移动操作** 等任务的 **USD 物体与 office/home/factory 场景** 开放包；README 描述完整 WWBench 含 **3K+ 刚体 / 60+ 铰链 / 软体与三套场景**，HF **v0.1 快照** 以 **cable + hinged 子集 + 三场景** 为主（约 **4.87 GB**）。

---

## 规模（README 愿景 vs HF v0.1 快照）

| 维度 | README（WWBench 总规划） | HF `Wuwen-AI/WWSimBench` v0.1（入库日树结构） |
|------|---------------------------|-----------------------------------------------|
| 刚体资产 | **3,000+** | v0.1 **未** 在 `objects/` 顶层单独列出 rigid 目录；以 README 为准，待后续 split 扩充 |
| 铰链资产 | **60+** | `objects/hinged_assets/` 四类目录合计 **~100** 个 USD 子文件夹（如 kitchen **63**、office device **11** 等） |
| 软体 | **20+** linear soft-body | v0.1 树中 **未见** 独立 soft-body 目录 |
| 线缆 / 插拔类 | 精度线缆插拔任务 | `objects/cable_assets/` **20** 个资产（含 `cable_*`、`defibrillator`、`iv_pole`、`chandelier` 等 USD） |
| 场景 | office / home / factory 三套 | `scenes/{office,home,factory}_scene/`（如 office 含 `desktop_task_scene.usd` 等） |
| HF 元数据 | — | **449** rows（Data Studio 预览）、**~4.87 GB** total |

## 目录结构（v0.1）

```
WWSimBench/
├── README.md
├── objects/
│   ├── cable_assets/          # 线缆、医疗设备、装饰等 USD
│   └── hinged_assets/
│       ├── desktop_digital_appliance/
│       ├── desktop_kitchen_appliance/
│       ├── desktop_office_device/
│       └── office_supporting_facility/
└── scenes/
    ├── office_scene/
    ├── home_scene/
    └── factory_scene/
```

单资产多为 **USD**（部分含 `SubUSDs/textures/`）；适合 **Isaac Sim / Omniverse** 等 USD 管线加载。

## 覆盖任务（README）

- 桌面 **tabletop manipulation**
- **铰链** 开合（hinge opening/closing）
- **线缆** 精度插拔（precision cable plugging and unplugging）
- **长距离移动操作**（long-distance mobile manipulation）

## 快速获取

```bash
pip install -U huggingface_hub
huggingface-cli download Wuwen-AI/WWSimBench --repo-type dataset --local-dir ./WWSimBench
```

或使用 `datasets` / `snapshot_download` 按子目录拉取 `objects/`、`scenes/`。

## 与无问 / 地平线生态

- 无问芯穹官网表述 **World Simulator + 仿真评测** 与地平线 **EmbodiedGen** 等合作做 **通用具身任务仿真评估**（见 [wuwen-ai-platform.md](../sites/wuwen-ai-platform.md)）。
- 资产定位接近 [EmbodiedGenData](./embodiedgen-data.md) 的 **sim-ready 环境层**，但 WWSimBench 更偏 **WWBench 评测资产命名与场景套件**，**不** 等价 EmbodiedGen 生成管线输出。

## 对 wiki 的映射

- 实体页：[WWSimBench](../../wiki/entities/wwsimbench.md)
- 交叉：[EmbodiedGen V2](../../wiki/entities/paper-embodiedgen-v2-sim-ready-world-engine.md)、[仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md)
