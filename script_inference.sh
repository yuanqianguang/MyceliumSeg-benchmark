export TMPDIR='/data1/yuanqianguang/tmp'

export CUDA_VISIBLE_DEVICES="0,1,2,3"

# ./dist_test.sh /data1/yuanqianguang/_mushroom/model/transformer_model/mycelium_mmseg_code/exp/multi_model/pspnet_unet_s5-d16.py /data1/yuanqianguang/_mushroom/model/transformer_model/mycelium_mmseg_code/work_dirs/pspnet_unet_s5-d16/iter_50000.pth 4 --eval mIoU mDice assd hd95 biou --tmpdir /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/tmp 

# ./dist_test.sh /data1/yuanqianguang/_mushroom/model/transformer_model/mycelium_mmseg_code/exp/multi_model/deeplabv3_r50-d8.py /data1/yuanqianguang/_mushroom/model/transformer_model/mycelium_mmseg_code/work_dirs/deeplabv3_r50-d8/iter_50000.pth 4 --eval mIoU mDice assd hd95 biou --tmpdir /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/tmp 

./dist_test.sh /data1/yuanqianguang/_mushroom/model/transformer_model/mycelium_mmseg_code/exp/multi_model/segformer.b0.1024x1024.mycelium.507.py /data1/yuanqianguang/_mushroom/model/transformer_model/mycelium_mmseg_code/work_dirs/segformer/iter_50000.pth 4 --eval mIoU mDice assd hd95 biou --tmpdir /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/tmp 
