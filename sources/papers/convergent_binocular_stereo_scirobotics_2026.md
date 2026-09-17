# Convergent Binocular Stereo: Depth Perception for Humanoid Robot Vision（Science Robotics 2026）

> 来源归档（ingest）

- **标题：** Convergent binocular stereo: Depth perception for humanoid robot vision
- **类型：** paper / stereo vision / active vision / humanoid perception / depth estimation
- **期刊：** Science Robotics, Vol. 11, Issue 117（2026-08-26）
- **DOI：** <https://doi.org/10.1126/scirobotics.aec7205>
- **PubMed：** <https://pubmed.ncbi.nlm.nih.gov/42647589/>
- **PDF：** <https://www.science.org/doi/pdf/10.1126/scirobotics.aec7205>
- **作者：** Mingshi Chi、John K. Tsotsos
- **机构：** 约克大学（York University）电气工程与计算机科学系；Mingshi Chi 亦隶属日本东北大学（Tohoku University）机器人系
- **关联学位论文：** [Convergent Active Stereo（MASc 2025）](https://yorkspace.library.yorku.ca/items/75b1c95d-8bc9-441f-84a2-8f740a9d65c1) — 算法与 CBS-BM 数据集前身
- **关联硬件：** [DIJIT: A Robotic Head for an Active Observer](https://arxiv.org/abs/2512.07998) — 同实验室主动双目头；Zenodo 说明链 [GitLab 标定仓](../repos/dijit-binocular-robotic-head.md)
- **代码 / 数据：** [Zenodo 10.5281/zenodo.21053380](../repos/convergent_binocular_stereo_zenodo.md)（MIT；`cbs-convergent-binocular-stereo.zip` + CBS-BM）
- **入库日期：** 2026-09-17
- **一句话说明：** 提出 **会聚双目立体（CBS）**：在主动会聚几何下同时估计水平/垂直视差，用 Gabor 粗到细 refinement；发布首个自然图像会聚立体基准 **CBS-BM**（49 场景、4-DoF 机器人采集）；与平行 SOTA 立体 broadly competitive，在重复纹理等场景更优——补全人形头「能 vergence 但不会算深度」的功能缺口。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 代码 + 数据集 | [Zenodo 21053380](https://doi.org/10.5281/zenodo.21053380) | `run_one.py` + `algo.py` + CBS-BM；MIT |
| DIJIT 标定仓 | [gitlab.nvision.eecs.yorku.ca/robots/dijit-binocular-robotic-head](https://gitlab.nvision.eecs.yorku.ca/robots/dijit-binocular-robotic-head) | 电机–相机标定；与 CBS 评测硬件同系 |
| 学位论文 | [YorkSpace MASc 2025](https://yorkspace.library.yorku.ca/items/75b1c95d-8bc9-441f-84a2-8f740a9d65c1) | 对角极线对应搜索、平行 vs 会聚系统对照 |
| 硬件论文 | [DIJIT arXiv:2512.07998](https://arxiv.org/abs/2512.07998) | 9 机械 + 4 光学 DoF；vergence/version/cyclotorsion |
| 补充图 | Zenodo `plots.zip`（~17 GB） | 手稿 Fig.6 直方图与表格复现脚本 |

## 摘要级要点

- **问题：** 人形机器人双目头常仿人眼做 **vergence / version**，但深度算法仍按 **平行双目** 假设设计；会聚几何下视差为 **二维向量场**（水平 + 垂直），且极线一般 **非水平**——现有 parallel stereo 管线无法直接复用。
- **CBS 算法：** 面向 **主动会聚双目机器人** 的解析立体算法；**Gabor 滤波响应 + 粗到细 refinement** 搜索对应点，显式估计 **水平与垂直视差**。
- **CBS-BM 基准：** 首个 **自然图像会聚立体** 数据集；**49 场景**、**4-DoF** 机器人系统采集；含平行配置图像/真值、各场景多 **fixation** 会聚视图、`fixations.csv` 电机与像素标注、`gt_disp.npy` 平行配置水平视差真值。
- **评测：** 平行 vs 会聚系统 **定量对照**；CBS 与 parallel SOTA ** broadly competitive**；在 **重复纹理** 场景及 **全场景平均水平视差/深度误差** 上 **优于** parallel 方法。
- **定位：** **不意图** 取代不需要类人行为的 parallel stereo；目标是让 **人形仿生头** 具备 **功能上 realistic** 的深度计算能力。

## 核心摘录（面向 wiki 编译）

### 1) 会聚 vs 平行几何（动机）

- **摘录要点：** 人类深度感知依赖两眼 **指向同一 fixation** 的会聚几何；部分机器人头虽有 vergence/version 但 **缺乏利用该几何的深度算法**；外观仿人 ≠ 运动满足会聚深度计算所需约束。
- **对 wiki 的映射：**
  - [Convergent Binocular Stereo 论文实体](../../wiki/entities/paper-convergent-binocular-stereo.md) — 问题陈述与 humanoid 约束

### 2) CBS 算法主干

- **摘录要点：** Gabor-filtered responses；coarse-to-fine refinement；同时输出 horizontal + vertical disparity；实现于 Zenodo 包 `algo.py`（内存密集，README 给优化 TODO）。
- **对 wiki 的映射：**
  - [Convergent Binocular Stereo 论文实体](../../wiki/entities/paper-convergent-binocular-stereo.md) — 核心原理
  - [立体匹配基础模型](../../wiki/methods/stereo-matching-foundation-models.md) — 与 parallel FM 路线对照

### 3) CBS-BM 数据结构

- **摘录要点：** 场景 0–48；每场景 `L_parallel_rect.png` / `R_parallel_rect.png`（已去畸变）、`gt_disp.npy`、`fixations/` 下 L/R 会聚图像、`motors/` 标定、`calib.yml` 内参（原分辨率，图像为 1/2 缩放需注意）。
- **对 wiki 的映射：**
  - [convergent_binocular_stereo_zenodo.md](../repos/convergent_binocular_stereo_zenodo.md) — 复现入口

### 4) 与 DIJIT 硬件线关系

- **摘录要点：** DIJIT 提供 vergence/version/cyclotorsion 与仿生扫视；CBS 提供 **会聚配置下的深度算法 + 基准**；Zenodo 复现需 DIJIT GitLab 电机标定。
- **对 wiki 的映射：**
  - [DIJIT 实体](../../wiki/entities/paper-notebook-dijit-a-robotic-head-for-an-active-observer.md)
  - [dijit-binocular-robotic-head.md](../repos/dijit-binocular-robotic-head.md)

### 5) 开源核查（步骤 2.5，2026-09-17）

- **已发布：** Zenodo [10.5281/zenodo.21053380](https://doi.org/10.5281/zenodo.21053380) — MIT；含 CBS 代码、`CBS-BM` 数据集、`plots/` 分析脚本；入口 `python3 run_one.py`。
- **已发布（关联）：** DIJIT 电机标定 [GitLab](https://gitlab.nvision.eecs.yorku.ca/robots/dijit-binocular-robotic-head)。
- **未列：** 无 GitHub 镜像；Science 项目页无独立 landing；真机 DIJIT 完整控制栈不在 CBS Zenodo 包内。
- **对 wiki 的映射：**
  - [Convergent Binocular Stereo 论文实体](../../wiki/entities/paper-convergent-binocular-stereo.md) — 工程实践 / 开源状态
