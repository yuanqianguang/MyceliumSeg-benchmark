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


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

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
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        
        # if args.eval:
        #     dataset.evaluate(outputs, args.eval, **kwargs)

        # if 'Hausdorff' not in args.eval:
        #     from medpy.metric.binary import hd95
        #     import numpy as np
        #     gts = list(dataset.get_gt_seg_maps())
        #     preds = [out.argmax(0).cpu().numpy() for out in outputs]
        #     hd_list = [hd95(p, g) for p, g in zip(preds, gts)]
        #     print(f'Mean Hausdorff Distance: {np.mean(hd_list):.4f}')

        # Extract specialized metrics from args.eval and store them separately
        # special_metrics = []
        # if args.eval:
        #     standard_metrics = []
        #     for metric in args.eval:
        #         if metric in ['ASSD', 'HD95', 'BIoU']:
        #             special_metrics.append(metric)
        #         else:
        #             standard_metrics.append(metric)
            
        #     if not standard_metrics:
        #         args.eval = None
        #     else:
        #         args.eval = standard_metrics
            
        #     print(f'Standard metrics: {args.eval}')
        #     print(f'Special metrics: {special_metrics}')

        # 1) 先跑 MMSeg 支持的所有内置指标
        if args.eval:
            print(f'Evaluating: {args.eval}')
            eval_results = dataset.evaluate(outputs, args.eval, **kwargs)
            print('>> Built-in metrics:', eval_results)





        # print(f'special_metrics:{special_metrics}')

        # # 2) 如果用户也要 ASSD，就手动算一遍
        # if 'ASSD' in special_metrics:
        #     print('carry ASSD')
        #     from mmseg.core.evaluation.hausdorff import safe_assd
        #     # 拿到所有 ground‑truth 和预测
        #     gts = list(dataset.get_gt_seg_maps())
        #     preds = [out.argmax(0).cpu().numpy() for out in outputs]
        #     hd_list = [safe_assd(p, g) for p, g in zip(preds, gts)]
        #     print(f'>> Mean Hausdorff Distance: {np.mean(hd_list):.4f}')

        # # 3) 如果用户也要 HD95，就手动算一遍
        # if 'HD95' in special_metrics:
        #     from mmseg.core.evaluation.hausdorff import safe_hd95
        #     # 拿到所有 ground‑truth 和预测
        #     gts = list(dataset.get_gt_seg_maps())
        #     preds = [out.argmax(0).cpu().numpy() for out in outputs]
        #     hd_list = [safe_hd95(p, g) for p, g in zip(preds, gts)]
        #     print(f'>> Max 95 Hausdorff Distance: {np.mean(hd_list):.4f}')

        # # 4) 如果用户也要 BIoU，就手动算一遍
        # if 'BIoU' in special_metrics:
        #     from mmseg.core.evaluation.biou import boundary_iou
        #     # 拿到所有 ground‑truth 和预测
        #     gts = list(dataset.get_gt_seg_maps())
        #     preds = [out.argmax(0).cpu().numpy() for out in outputs]
        #     hd_list = [boundary_iou(g, p) for p, g in zip(preds, gts)]
        #     print(f'>> Boundary-IoU Distance: {np.mean(hd_list):.4f}')

if __name__ == '__main__':
    main()
