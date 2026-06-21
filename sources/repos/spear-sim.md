# SPEAR (spear-sim/spear)

> 来源归档

- **标题：** SPEAR: A Simulator for Photorealistic Embodied AI Research
- **类型：** repo
- **来源：** spear-sim（GitHub 组织）
- **链接：** https://github.com/spear-sim/spear
- **Stars / Forks：** ~433 / —（2026-06）
- **许可证：** 代码 MIT；资产 CC0
- **入库日期：** 2026-06-21
- **一句话说明：** 面向具身 AI 与合成视觉的 **Unreal Engine 可编程仿真库**：Python 经插件连接任意 UE 应用，暴露 14K+ 反射 API、56 FPS 级 1080p 光真实感渲染与 Hypersim 级 GT 模态，并以 `begin_frame`/`end_frame` 事务模型在单帧内确定性执行复杂 UE 工作图。
- **沉淀到 wiki：** 是 → [`wiki/entities/spear-sim.md`](../../wiki/entities/spear-sim.md)

---

## 核心定位

**SPEAR** 不是「自带机器人与 RL 环境的游戏式仿真器」，而是 **把任意 Unreal Engine 项目变成可脚本化研究后端** 的 Python 库：

- 通过模块化插件架构 **启动或附着** 到现有 UE 可执行程序；
- 凡 UE 反射系统可见的 C++ 函数/属性（`UFUNCTION` / `UPROPERTY`）均可从 Python 调用；
- 官方称相较既有 UE 类仿真器，可编程面扩大约 **一个数量级**（**14K+** 独立 UE 函数）。

论文定位（ECCV 2026）：解决现有光真实感仿真器在 **通用性、可编程性、渲染速度** 三方面的瓶颈，服务具身智能训练与合成视觉数据生成。

---

## 关键能力摘要

| 能力 | 说明 |
|------|------|
| **渲染吞吐** | 单实例 1920×1080 beauty 帧直写 NumPy，约 **56 FPS**；官方称较既有 UE 插件快约一个数量级 |
| **GT 模态** | 深度、法线、实例/语义 ID、**非漫反射内禀分解**、材质 ID、基于物理的着色参数等；覆盖 Hypersim 数据集模态并扩展 |
| **编程模型** | `instance.begin_frame()` / `end_frame()` 配对：同帧内先写控制、末帧读观测；支持 `call_async.*` 非阻塞 UE 调用 |
| **多智能体 / 多动作空间** | 示例：CitySample 人/车、StackOBot 飞行机器人、CropoutSample 多体、GameAnimationSample 跑酷人形与四足 |
| **PCG / 场景** | 可脚本驱动 UE 程序化内容生成（如 ElectricDreams 岩石结构平移时水体/原木自适应） |
| **Co-simulation** | 与 **MuJoCo** 交互：MuJoCo viewer 施力，实时查询状态并由 SPEAR 同步 UE 场景 |
| **数字人** | MetaHumans 样例：同步多视角高细节人脸渲染 |
| **NL 场景编辑** | 视觉-语言 coding assistant 写 SPEAR 程序响应文本提示（agentic scene editing 示例） |

---

## 最小程序概念（README 摘录）

```python
config = spear.get_config(user_config_files=["user_config.yaml"])
instance = spear.Instance(config=config)
game = instance.get_game()

with instance.begin_frame():
    bp_axes_uclass = game.unreal_service.load_class(
        uclass="AActor", name="/SpContent/Blueprints/BP_Axes.BP_Axes_C"
    )
    bp_axes = game.unreal_service.spawn_actor(
        uclass=bp_axes_uclass, location={"X": -10.0, "Y": 280.0, "Z": 50.0}
    )
    bp_axes.SetActorScale3D(NewScale3D={"X": 4.0, "Y": 4.0, "Z": 4.0})
    future = bp_axes.RootComponent.get().call_async.K2_GetComponentLocation()

with instance.end_frame():
    location = future.get()
```

扩展 API：在 C++ 头文件为函数/变量添加 `UFUNCTION` / `UPROPERTY` 即可暴露给 Python。

---

## 文档与示例入口

| 资源 | 链接 |
|------|------|
| Getting Started | `docs/getting_started.md` |
| Example Applications | `docs/running_our_example_applications.md` |
| Import/Export Assets | `docs/importing_and_exporting_assets.md` |
| Natural Language Control | `docs/controlling_with_natural_language.md` |

---

## 与本库其它页面的关系

| 资料 | 关系 |
|------|------|
| [airsim.md](airsim.md) | 同为 UE 视觉仿真；AirSim 偏 UAV/AD 与维护期项目，SPEAR 强调 **通用 UE 反射 API + 高速 GT 渲染** |
| [mujoco.md](../repos/mujoco.md) | SPEAR 官方示例 **MuJoCo↔UE 协同仿真**；物理环可在 MuJoCo，视觉/场景在 UE |
| [metahuman 站点](../sites/metahuman-com.md) | SPEAR 演示 MetaHumans 多视角相机；数字人资产与 SPEAR 相机传感器互补 |
| [isaac-gym-isaac-lab](../../wiki/entities/isaac-gym-isaac-lab.md) | Isaac 栈主打 **GPU 并行 RL 环境**；SPEAR 主打 **任意 UE 项目的光真实感可编程后端**，并行训练生态尚非主线 |

---

## 对 wiki 的映射

- **新建实体页：** [`wiki/entities/spear-sim.md`](../../wiki/entities/spear-sim.md) — SPEAR 定位、编程模型、GT 模态、与 AirSim / Isaac / MuJoCo 的分工
- **交叉更新：** [`wiki/queries/simulator-selection-guide.md`](../../wiki/queries/simulator-selection-guide.md)（UE 光真实感 / 合成数据分支）、[`wiki/entities/metahuman.md`](../../wiki/entities/metahuman.md)、[`wiki/entities/airsim.md`](../../wiki/entities/airsim.md)

---

## 引用

```bibtex
@inproceedings{roberts:2026,
    author = {Mike Roberts and Renhan Wang and Rushikesh Zawar and Rachith Prakash
              and Quentin Leboutet and Stephan Richter and Matthias M{\"u}ller
              and German Ros and Rui Tang and Stefan Leutenegger and Yannick
              Hold-Geoffroy and Kalyan Sunkavalli and Vladlen Koltun},
    title = {{SPEAR}: {A} Simulator for Photorealistic Embodied {AI} Research},
    booktitle = {ECCV 2026}
}
```
