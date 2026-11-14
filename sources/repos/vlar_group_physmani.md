# vLAR-group/PhysMani

> 来源归档

- **标题：** PhysMani（官方仓库 / 项目页）
- **类型：** repo
- **组织：** vLAR Group（香港理工大学 PolyU）
- **代码：** <https://github.com/vLAR-group/PhysMani>
- **论文：** <https://arxiv.org/abs/2607.01938>（PDF：<https://arxiv.org/pdf/2607.01938v1>）
- **venue：** ECCV 2026
- **许可证：** CC BY-NC-SA 4.0
- **入库日期：** 2026-07-14
- **一句话说明：** **PhysMani** 官方 GitHub 入口：physics-principled **3D Gaussian 世界模型** + future-aware **3DFA 策略** 的动态操作框架；含 **PhysMani-Bench（16 任务）** 说明。截至 2026-10-04，官方 pre-release 已含 PhysMani 训练/仿真评测代码、PhysMani-Bench 数据下载脚本及 checkpoint 下载脚本；数据与发布 checkpoint 见 [Hugging Face](https://huggingface.co/datasets/vLAR/PhysMani-Bench)。比较基线与重构版代码结构暂未打包。

## Release Status（截至 2026-10-04）

| 组件 | 状态 |
|------|------|
| README / License / 引用 | 已公开 |
| PhysMani 训练与仿真评测代码 | 已公开（pre-release；通过官方仓库及锁定的 submodules 运行） |
| PhysMani-Bench 数据 | 已公开下载入口：[Hugging Face](https://huggingface.co/datasets/vLAR/PhysMani-Bench) |
| 发布 checkpoint | 已公开下载脚本与 HF 数据集入口 |
| 3DDA / ManiGaussian / π0 等比较基线 | 尚未纳入当前 release |
| 重构版代码结构 | 尚未发布 |

训练默认使用 3 GPU；Docker runtime 会编译 CUDA 扩展。第三方 submodule 的依赖与版本见 [reproducibility/versions.json](https://github.com/vLAR-group/PhysMani/blob/main/reproducibility/versions.json)。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [PhysMani（动态操作 3D 世界模型）](../../wiki/entities/paper-physmani-dynamic-manipulation-world-model.md) | 实体归纳页：双模块架构、Benchmark、仿真/真机结果 |
| [Generative World Models](../../wiki/methods/generative-world-models.md) | **3DGS 速度场 WM** 相对 **2D 视频扩散 WM** 的机器人控制向样本 |
| [Manipulation](../../wiki/tasks/manipulation.md) | **动态目标**（传送带、旋转架等）操作评测语境 |
| [Kairos](../../wiki/entities/paper-kairos-native-world-model-stack.md) | 同为 **世界模型 + 动作策略** 联合；Kairos 走 **视频 DiT WAM**，PhysMani 走 **3D Gaussian 物理速度场 + 3DFA IL** |

## 对 wiki 的映射

- 技术报告摘录：[`sources/papers/physmani_arxiv_2607_01938.md`](../papers/physmani_arxiv_2607_01938.md)
- 沉淀 **[`wiki/entities/paper-physmani-dynamic-manipulation-world-model.md`](../../wiki/entities/paper-physmani-dynamic-manipulation-world-model.md)**

## 固定版本源码复核（2026-11-13）

- 版本：[b9eb1ef8a40dbb80b1b0383b287a689983c83918](https://github.com/vLAR-group/PhysMani/tree/b9eb1ef8a40dbb80b1b0383b287a689983c83918)。已读根README、policy/world server启动sh全文；WorldModelServer.infer/_handle_update/_handle_predict、WorldModel.update/predict/predict_velmappc、PolicyServer.infer、Actioner.predict，ForesightDiffuserActorV6的encode_inputs/推理forward与VelocityFeatureModuleV2.forward完整路径。
- v1.0已重构为physmani_bench/policy/wm三个本仓包，3DFA基线也已发布；旧「landing page」「重构未提供」失效。3DDA/ManiGaussian/π0.5基线仍TODO；CC BY-NC-SA4.0及第三方许可分别核对。
- WM服务分init/update/predict；update依次train_network_one_step、train_gaussian_one_step、transform_gaussian_to；velocity模式返回velmappc_list。policy.infer把velmap_pc透传模型；encode_inputs归一化位置，邻域VelocityFeatureModuleV2用cdist/topk、距离mask及注意力与视觉特征融合，再走继承的轨迹生成。
- 当前默认单任务gpu0.sh为checkpoint99999；8-GPU全套16任务640episode，CPU/NUMA绑定依硬件，不能照旧写100episode/任务。未安装Docker依赖、下载数据权重或实测。
