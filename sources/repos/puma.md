# PUMA（PRBonn/puma）

- **URL：** <https://github.com/PRBonn/puma>
- **论文：** [Poisson Surface Reconstruction for LiDAR Odometry and Mapping](../papers/puma_icra_2021_vizzo.md)（ICRA 2021）
- **作者页：** <https://www.ipb.uni-bonn.de/people/ignacio-vizzo/index.html>
- **视频：** <https://youtu.be/7yWtYWaO5Nk>

## 一句话说明

Poisson 曲面重建 LiDAR 里程计与建图：三角 mesh 地图 + ray casting scan-to-mesh + point-to-plane 配准。

## 主要入口（README）

| 路径 | 作用 |
|------|------|
| `apps/pipelines/slam/puma_pipeline.py` | 端到端 SLAM 管线 |
| `apps/pipelines/odometry/icp_frame_2_mesh.py` | frame-to-mesh 配准 |
| `apps/run_poisson.py` | Poisson 重建 |
| `apps/data_conversion/bin2ply.py` | KITTI `.bin` → `.ply` |
| `docker/` + `Makefile` | 推荐 Docker 运行环境 |

## 运行要点

```sh
export DATASETS=<full-path-to-datasets>
make   # 构建 apss docker 镜像
docker-compose run --rm apps bash -c './pipelines/slam/puma_pipeline.py --dataset ./data/kitti-odometry/ply --sequence 07 --n_scans 40'
```

输出：`results/*_p2l_raycasting.ply`（mesh）、`.txt`（位姿）、`.yml`（配置）。

## 交叉链接

- [paper-puma-lidar-mesh-odometry 论文实体](../../wiki/entities/paper-puma-lidar-mesh-odometry.md)
- [IPB 作者页归档](../sites/ipb-puma-vizzo.md)
- [论文归档](../papers/puma_icra_2021_vizzo.md)
