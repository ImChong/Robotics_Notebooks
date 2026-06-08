# PHC（仓库）

> 来源归档（工具向；论文策展见 `sources/papers/bfm_awesome_phc_arxiv_2305_06456.md`）

- **标题：** PHC: Perpetual Humanoid Control
- **类型：** repo
- **链接：** https://github.com/ZhengyiLuo/PHC
- **项目页：** https://www.zhengyiluo.com/PHC-Site/
- **论文：** ICCV 2023，arXiv:2305.06456
- **入库日期：** 2026-06-08
- **一句话说明：** 物理人形控制与大规模动作模仿框架；提供 SMPL→自定义人形的 shape/motion fitting 重定向脚本（`docs/retargeting.md`），常与 GMR / OmniRetarget 作基线对比。
- **沉淀到 wiki：** 是 → [`wiki/entities/phc.md`](../../wiki/entities/phc.md)

## 重定向相关入口

- `python scripts/data_process/fit_smpl_shape.py robot=<your_robot>`
- `python scripts/data_process/fit_smpl_motion.py robot=<your_robot> +amass_root=...`
- YAML：`joint_matches`、`smpl_pose_modifier`、`extend_config`
