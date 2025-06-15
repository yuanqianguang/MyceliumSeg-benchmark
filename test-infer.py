import argparse
import os

import mmcv
import torch

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
# mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from IPython import embed

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', default='work_dirs/res.pkl', help='output result file in pickle format')
    # 新增保存预测图的参数
    parser.add_argument('--save-pred', action='store_true', help='Save prediction results as images')
    parser.add_argument('--pred-dir', default='work_dirs/predictions', help='directory to save prediction images')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default='mIoU',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def save_predictions(outputs, dataset, pred_dir):
    """保存预测结果为图像"""
    import cv2
    from PIL import Image
    
    os.makedirs(pred_dir, exist_ok=True)
    print(f'Saving predictions to {pred_dir}')
    print(f'Loading predictions from {len(outputs)} temp files...')
    
    for i, output_path in enumerate(outputs):
        try:
            # 从临时文件加载预测结果
            if isinstance(output_path, str) and output_path.endswith('.npy'):
                pred_data = np.load(output_path)
                print(f"Loaded prediction {i}: shape {pred_data.shape}, dtype {pred_data.dtype}")
                
                # 处理预测数据
                if len(pred_data.shape) == 3:  # [C, H, W] - logits
                    pred = pred_data.argmax(0).astype(np.uint8)
                elif len(pred_data.shape) == 2:  # [H, W] - already class predictions
                    pred = pred_data.astype(np.uint8)
                else:
                    print(f"Unexpected prediction shape: {pred_data.shape}")
                    continue
                    
            else:
                print(f"Unexpected output format: {output_path}")
                continue
            
            # 获取文件名
            if i < len(dataset.img_infos):
                img_info = dataset.img_infos[i]
                img_name = os.path.basename(img_info['filename']).split('.')[0]
            else:
                img_name = f"pred_{i:06d}"
            
            # 保存原始预测图 (灰度图)
            pred_path = os.path.join(pred_dir, f'{img_name}_pred.png')
            cv2.imwrite(pred_path, pred)
            print(f"Saved grayscale prediction: {pred_path}")
            
            # 如果有调色板，也保存彩色版本
            if hasattr(dataset, 'PALETTE') and dataset.PALETTE is not None:
                # 创建彩色预测图
                colored_pred = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                palette = dataset.PALETTE
                
                for class_id in range(min(len(palette), pred.max() + 1)):
                    mask = pred == class_id
                    if mask.any():
                        colored_pred[mask] = palette[class_id]
                
                colored_path = os.path.join(pred_dir, f'{img_name}_pred_colored.png')
                cv2.imwrite(colored_path, cv2.cvtColor(colored_pred, cv2.COLOR_RGB2BGR))
                print(f"Saved colored prediction: {colored_path}")
                
        except Exception as e:
            print(f"Error processing prediction {i}: {e}")
            continue
    
    print(f'Finished saving prediction images to {pred_dir}')

def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir or args.save_pred, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show", "--show-dir" or "--save-pred"')

    if 'None' in args.eval:
        args.eval = None
    if args.eval and args.format_only:

        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        if cfg.data.test.type == 'CityscapesDataset':
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
            ]
            cfg.data.test.pipeline[1].flip = True
        elif cfg.data.test.type == 'ADE20KDataset':
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.75, 0.875, 1.0, 1.125, 1.25
            ]
            cfg.data.test.pipeline[1].flip = True
        else:
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75
            ]
            cfg.data.test.pipeline[1].flip = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    print('cfg.data.test:', cfg.data.test)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']

    efficient_test = True #False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  efficient_test)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
                
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, efficient_test)

    rank, _ = get_dist_info()
    if rank == 0:
        # 先调试输出格式
        print(f"Outputs length: {len(outputs)}")
        if len(outputs) > 0:
            print(f"First output type: {type(outputs[0])}")
            print(f"First output content: {outputs[0]}")
            if hasattr(outputs[0], 'shape'):
                print(f"First output shape: {outputs[0].shape}")
            input('look at the first output, press Enter to continue...')
                
        
        # 保存预测图像
        if args.save_pred:
            save_predictions(outputs, dataset, args.pred_dir)
        
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        
        # 1) 先跑 MMSeg 支持的所有内置指标
        if args.eval:
            print(f'Evaluating: {args.eval}')
            eval_results = dataset.evaluate(outputs, args.eval, **kwargs)
            print('>> Built-in metrics:', eval_results)

if __name__ == '__main__':
    main()
