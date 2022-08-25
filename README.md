An pointcloud segmentation tools implemented in paddlepaddle.

Download 3D indoor parsing dataset (**S3DIS**) [here](http://buildingparser.stanford.edu/dataset.html)  and save in `data/s3dis/Stanford3dDataset_v1.2_Aligned_Version/`.
```
python data_prepare/collect_indoor3d_data.py
```

train command:
```commandline
python tools/train.py --config configs/pointnet/pointnet_24k_200e.yml --use_vdl --log_iter 10 --save_interval 1 --save_dir output --do_eval --num_workers 6
```
