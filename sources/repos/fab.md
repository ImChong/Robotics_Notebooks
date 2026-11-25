# FAB 官方源码

- **类型：** repo
- **仓库：** https://github.com/noahcao/FAB
- **论文：** https://arxiv.org/abs/2611.25801
- **项目：** https://www.jinkuncao.com/FAB/
- **权重：** https://huggingface.co/noahcao/FAB
- **版本：** pyproject.toml 声明 1.0.0，main 核对于 2026-11-24
- **许可：** 主体 Apache-2.0，SAM/DINOv3 衍生代码及依赖另有条款；权重非商业研究用途
- **核对范围：** README、pyproject.toml、fab/entry.py、fab/cli.py、scripts/fab_world.sh 的调用链及 scripts/fab_retarget.py 的输入与入口部分；未执行安装、推理或仿真。

## 运行入口

安装 Python 3.10+、PyTorch 2.4+，按 README 安装 retarget/world extras，下载 FAB、SAM配置/MHR、SMPL及G1描述。

```bash
fab assets/demo/skater_seated.jpg --out output/fab
fab-video assets/demo/dancer_street.mp4 -o output/dancer.mp4
fab-world assets/demo/steps_up.mp4 --out output/world
fab-retarget output/world/steps_up.npz --out output/g1
```

- `fab` → fab.cli:main → FABPredictor.predict → pickle。
- `fab-world` → fab.entry:world → scripts/fab_world.sh → bench.world_vipe、bench.world_geocalib、bench.videomimic.recon、bench.scene_recon.mvs_depth、bench.scene_recon.joint → 世界NPZ、场景PLY和高度场。
- `fab-retarget` → fab.entry:retarget → scripts/fab_retarget.py；读取世界结果及中间稠密点，支持points/mesh/skeleton；输出50 FPS G1根位姿与关节轨迹NPZ及JSON。
- points模式若缺FAB原始稠密点，脚本会提示并使用人体网格相应顶点，不能将其视为原始点预测。
- [论文归档](../papers/fab_arxiv_2611_25801.md)
- [项目页归档](../sites/fab.md)
- [知识实体](../../wiki/entities/paper-fab.md)
