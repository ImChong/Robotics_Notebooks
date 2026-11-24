# DecomVoxel 官方实现

- **地址：** https://github.com/DecomVoxel/DecomVoxel
- **论文：** https://arxiv.org/abs/2610.01914
- **项目：** https://decomvoxel.github.io/DecomVoxel-Webpage/
- **数据：** https://huggingface.co/datasets/JunfengNi/DecomVoxel
- **核对日期：** 2026-11-24
- **核对范围：** 官方 README、run_replica_demo.sh、train_demo.py 的导入、入口及 run_full_pipeline 后半段；内部 SDS 模块尚待逐行核对，未执行 GPU demo。
- **通读：** 未完成

## 复现入口

```bash
source env.sh
bash run_replica_demo.sh
```

脚本将 Replica/scan1 输入传入 train_demo.py --mode full_pipeline，默认 200 views、3 workers，使用 configs/replica_config.yaml，输出 outputs/demo_replica/scan1。README 另要求 Python 3.10 环境、外部模型权重和条件图生成服务密钥（Seedream 默认需要 ARK_API_KEY；其他两种适配器需要 ATLASCLOUD_API_KEY）。不得把密钥写入仓库。项目公开了数据入口，但未实际下载；根目录未见独立 LICENSE 文件，授权范围需要另核实。

## 已核对的入口职责

- train_demo.py：解析 CLI，选择 GPU，run_full_pipeline 串接五阶段。
- run_train_geosvr / run_segm_3d：场景重建及实例分割。
- generate_cond_img_avo_para：按对象生成条件图，排除背景 object_000。
- run_scene_sds_para：接收 semantic_result/object_voxels 与条件图，输出 scene_sds。
- run_scene_appearance_sds_para：按对象执行外观优化。
- 已确认导入 score_distillation_sampling_ss.run_sds_completion_ss 和 score_distillation_sampling_appearance 的 encode_object_slat、slat_sds、decode_slat_to_mesh；具体内部调用尚待核查。

背景补全另有 bg_inpaint_replica.sh，不能假定由 full_pipeline 自动执行。

## 对 wiki 的映射

- [论文归档](../papers/decomvoxel_arxiv_2610_01914.md)
- [项目归档](../sites/decomvoxel.md)
- [DecomVoxel](../../wiki/entities/paper-decomvoxel.md)
