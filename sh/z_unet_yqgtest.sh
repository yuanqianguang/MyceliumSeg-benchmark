export TMPDIR='/data1/yuanqianguang/tmp'

export CUDA_VISIBLE_DEVICES="2,3,4,5"

# ./tools/dist_test.sh l/data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/local_configs/segformer/B0/segformer.b0.4608x3456.mycelium.507.py /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/work_dirs/0416segformer.b0.4608x3456.mycelium.507/iter_9125.pth 1 --eval mIoU 


# python test.py /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/local_configs/segformer/B0/segformer.b0.4608x3456.mycelium.507.py /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/work_dirs/segformer.b0.4608x3456.mycelium.507/iter_9125.pth --eval mIoU mDice assd hd95 biou

# python test.py /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/local_configs/segformer/B0/segformer.b0.1024x1024.mycelium.507.py /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/work_dirs/segformer.b0.1024x1024.mycelium.507/iter_83750.pth --eval mIoU mDice assd hd95 biou



# python test.py /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/local_configs/segformer/B0/segformer.b0.1024x1024.mycelium.507.py /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/work_dirs/segformer.b0.1024x1024.mycelium.507/iter_83750.pth --eval mIoU mDice assd hd95 biou --tmpdir /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/tmp 


# ./dist_test.sh /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/local_configs/segformer/B0/segformer.b0.1024x1024.mycelium.507.py /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/work_dirs/segformer.b0.1024x1024.mycelium.507/iter_83750.pth 4 --eval mIoU mDice assd hd95 biou --tmpdir /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/tmp 

# ./dist_test.sh /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/local_configs/segformer/B0/segformer.b0.1024x1024.mycelium.507.py /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/work_dirs/segformer.b0.1024x1024.mycelium.507/iter_50000.pth 4 --eval mIoU mDice assd hd95 biou --tmpdir /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/tmp 


# ./dist_test.sh /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/mycelium_model/model/deeplabv3_unet_s5-d16.py /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/work_dirs/deeplabv3_unet_s5-d16/iter_50000.pth 4 --eval mIoU mDice assd hd95 biou --tmpdir /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/tmp 

# ./dist_test.sh /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/mycelium_model/model/fcn_unet_s5-d16.py /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/work_dirs/fcn_unet_s5-d16/iter_50000.pth 4 --eval mIoU mDice assd hd95 biou --tmpdir /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/tmp 

./dist_test.sh /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/mycelium_model/model/pspnet_unet_s5-d16.py /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/work_dirs/pspnet_unet_s5-d16/iter_50000.pth 4 --eval mIoU mDice assd hd95 biou --tmpdir /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/tmp 