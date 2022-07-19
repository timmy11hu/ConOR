import os
import argparse
import logging
import warnings
import sys
import time
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# running in parent dir
os.chdir("..")
sys.path.append(".")
print("Current Working Directory ", os.getcwd())

from dp.utils.average_meter import AverageMeter
from dp.utils.config import load_config, print_config
from dp.utils.pyt_io import create_summary_writer
from dp.core.solver import Solver
from dp.datasets.loader import build_loader
from dp.utils.test_visualizer import Visualizer
from dp.utils.evaluator import Metrics

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--config', type=str, )
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--loss_type', default='conor', type=str, help='conor,or,gl,lgl,bc,mcc')
parser.add_argument('--epoch', default=10, type=int)
args = parser.parse_args()


if not args.config:
    logging.error('args --config should be available.')
    raise ValueError
is_main_process = True if args.local_rank == 0 else False

config = load_config(args.config)
config['environ']['seed'] = args.seed
config['model']['params']['loss_type'] = args.loss_type
config['solver']['epochs'] = args.epoch

solver = Solver()
solver.init_from_scratch(config)

if is_main_process:
    print_config(config)
    exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    snap_dir = os.path.join(config["snap"]["path"], config['data']['name'],
                            config["model"]["params"]["loss_type"], exp_time)
    if not os.path.exists(snap_dir):
        os.makedirs(snap_dir)
    writer = create_summary_writer(snap_dir)
    visualizer = Visualizer(config, writer)

# dataset
tr_loader, sampler, niter_per_epoch = build_loader(config, True, solver.world_size, solver.distributed)
te_loader, _, niter_test = build_loader(config, False, solver.world_size, solver.distributed)

# metric
loss_meter = AverageMeter()
metric = Metrics()
metric.reset()
best_rmse = 10.0

"""
    usage: debug
"""
# niter_per_epoch, niter_test = 1, 10


for epoch in range(solver.epoch + 1, config['solver']['epochs'] + 1):
    solver.before_epoch(epoch=epoch)
    if solver.distributed:
        sampler.set_epoch(epoch)

    if is_main_process:
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(niter_per_epoch), file=sys.stdout, bar_format=bar_format)
    else:
        pbar = range(niter_per_epoch)
    loss_meter.reset()
    train_iter = iter(tr_loader)
    for idx in pbar:
        minibatch = train_iter.next()
        filtered_kwargs = solver.parse_kwargs(minibatch)
        loss = solver.step(filtered_kwargs['image'], filtered_kwargs['target'], filtered_kwargs.get('weight'))
        loss_meter.update(loss)

        if is_main_process:
            print_str = '[Train] Epoch{}/{}'.format(epoch, config['solver']['epochs']) \
                        + ' Iter{}/{}:'.format(idx + 1, niter_per_epoch) \
                        + ' lr=%.8f' % solver.get_learning_rates()[0] \
                        + ' losses=%.2f' % loss.item() \
                        + '(%.2f)' % loss_meter.mean() \

            pbar.set_description(print_str, refresh=False)

    solver.after_epoch()


    # Validation
    if is_main_process:
        pbar = tqdm(range(niter_test), file=sys.stdout, bar_format=bar_format)
    else:
        pbar = range(niter_test)
    metric.reset()
    test_iter = iter(te_loader)
    for idx in pbar:
        minibatch = test_iter.next()
        filtered_kwargs = solver.parse_kwargs(minibatch)
        pred, variance = solver.step_no_grad(filtered_kwargs['image'])
        metric.compute_metric(pred=pred, target=filtered_kwargs['target'], uncertainty=variance)

        if is_main_process:
            print_str = '[Test] Epoch{}/{}'.format(epoch, config['solver']['epochs']) \
                        + ' Iter{}/{}: '.format(idx + 1, niter_test) \
                        + metric.get_snapshot_info() \

            pbar.set_description(print_str, refresh=False)
        """
        visualization for model output and feature maps.
        """
        if is_main_process and idx % 10 == 0:
            visualizer.visualize(minibatch, pred, epoch=epoch, y_var=variance)

    # Save
    if is_main_process:
        logging.info('After Epoch{}/{}, {}'.format(epoch, config['solver']['epochs'], metric.get_result_info()))
        writer.add_scalar("Train/loss", loss_meter.mean(), epoch)
        metric.add_scalar(writer, tag='Test', epoch=epoch)

        current_rmse = metric.rmse.mean()[0]
        if current_rmse < best_rmse:
            snap_name = os.path.join(snap_dir, 'epoch-{}.pth'.format(epoch))
            # solver.save_checkpoint(snap_name)
            best_rmse = current_rmse

if is_main_process:
    writer.close()
