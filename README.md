# README

我们实现了GeoSync，实现对几何攻击的鲁棒性与水印设计的解耦，通过编码局部水印信息、检测几何变换并反转的方法增强水印在几何攻击的鲁棒性，并使得更多的水印设计成为可能。我们在Tree Ring上实验了我们的框架。

## 环境

创建环境：

```bash
conda create -n geosync python=3.12
```

激活环境

```bash
conda activate geosync
```

安装依赖

```bash
pip install -e .
pip install -r requirements.txt
```

## 运行

你可能需要下载子模块：

```bash
git submodule init
git submodule update
```

创建链接

```bash
ln -s deps/wam/configs/ configs
```

下载模型

```bash
./scripts/download.sh
```

可以使用脚本[tree_ring_all.sh](./scripts/tree_ring_all.sh)运行我们的方法：

```bash
# ./scripts/tree_ring_all.sh <if-use-geosync> <if-use-imagenet> watermark-pattern sync-type
# GeoSync imagenet
./scripts/tree_ring_all.sh true true ring sync_seal
./scripts/tree_ring_all.sh true true rand sync_seal
# GeoSync Stable Diffusion
./scripts/tree_ring_all.sh true false ring sync_seal
./scripts/tree_ring_all.sh true false rand sync_seal
# GeoSync using WAM as synchronization
./scripts/tree_ring_all.sh true true ring wam
./scripts/tree_ring_all.sh true true rand wam
./scripts/tree_ring_all.sh true false ring wam
./scripts/tree_ring_all.sh true false rand wam
# Original Tree Ring
./scripts_tree_ring_all.sh false true ring
./scripts_tree_ring_all.sh false true rand
./scripts_tree_ring_all.sh false false ring
./scripts_tree_ring_all.sh false false rrand
```

<!-- 下载 SyncSeal 模型：
```bash
wget -P checkpoints https://dl.fbaipublicfiles.com/wmar/syncseal/paper/syncmodel.jit.pt
```
目前只做了 imagenet 256 × 256 模型上的代码修改（run_tree_ring_watermark_imagenet.py），运行方式见 scripts/tree_ring_imagenet.sh -->