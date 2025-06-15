# python tools/train.py <CONFIG_FILE> --options model.pretrained=<PRETRAIN_MODEL> 
# tools/dist_train.sh /root/autodl-tmp/___/Swin-Transformer-Semantic-Segmentation/configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py 1 --options model.pretrained=/root/autodl-tmp/___/Swin-Transformer-Semantic-Segmentation/swin_tiny_patch4_window7_224.pth

# tools/dist_train.sh configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py 1 --options model.pretrained=./swin_tiny_patch4_window7_224.pth
export TMPDIR='/data1/yuanqianguang/tmp'

export CUDA_VISIBLE_DEVICES="0,1,2,3"
# bash dist_train.sh /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/local_configs/segformer/B0/segformer.b0.4608x3456.mycelium.507.py 4
bash dist_train.sh /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/mycelium_model/model/deeplabv3_unet_s5-d16.py 4

bash dist_train.sh /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/mycelium_model/model/fcn_unet_s5-d16.py 4

bash dist_train.sh /data1/yuanqianguang/_mushroom/model/transformer_model/SegFormer-master/mycelium_model/model/pspnet_unet_s5-d16.py 4
