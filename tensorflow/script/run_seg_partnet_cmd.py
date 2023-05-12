import os
import csv
import math
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, required=False,
                    default='0811_partnet_randinit')
parser.add_argument('--gpu', type=int, required=False, default=0)
parser.add_argument('--mode', type=str, required=False, default='randinit')
parser.add_argument('--ckpt', type=str, required=False,
                    default='dataset/midnet_data/mid_d6_o6/model/iter_800000.ckpt')


args = parser.parse_args()
alias = args.alias
gpu = args.gpu
mode = args.mode


factor = 2
batch_size = 32
ckpt = args.ckpt if mode != 'randinit' else '\'\''
module = 'run_seg_partnet_finetune.py' if mode != 'randinit' else 'run_seg_partnet.py'
script = f'python {module} --config configs/seg_hrnet_partnet_pts.yaml'
if mode != 'randinit':
  script += f' SOLVER.mode {mode} '
data = 'dataset/partnet_segmentation/dataset'


names     = ['Bed', 'Bottle', 'Chair', 'Clock', 'Dishwasher', 'Display', 'Door', 
             'Earphone', 'Faucet', 'Knife', 'Lamp', 'Microwave', 'Refrigerator',
             'StorageFurniture', 'Table', 'TrashCan', 'Vase']
train_num = [ 133,   315,  4489,  406,   111,   633,   149,  147,  
              435,   221,  1554,  133,   136,  1588,  5707,  221,  741]
max_iters = [3000,  3000, 20000, 5000,  3000,  5000,  3000, 3000, 
             5000,  3000, 10000, 3000,  3000, 10000, 20000, 3000, 10000]
test_iters= [ 100,   100,   800,  400,   200,   400,   200,  200, 
              400,   200,   800,  200,   200,   800,   800,  200,   800]
test_num  = [  37,    84,  1217,   98,    51,   191,    51,   53, 
              132,    77,   419,   39,    31,   451,  1668,   63,   233]
val_num   = [  24,    37,   617,   50,    19,   104,    25,   28,   
               81,    29,   234,   12,    20,   230,   843,   37,   102]
seg_num   = [  15,     9,    39,   11,     7,     4,     5,   10,  
               12,    10,    41,    6,     7,    24,    51,   11,     6]
ratios    = [0.01,  0.02,  0.05, 0.10,  0.20,  0.50, 1.00]
muls      = [   2,     2,     2,    1,     1,     1,    1]  # longer iter when data < 10%


for i in range(len(ratios)-1, -1, -1):
  for k in range(len(names)):
    ratio, name = ratios[i], names[k]
    max_iter = int(max_iters[k] * ratio * muls[i])
    step_size1, step_size2 = int(0.5 * max_iter), int(0.25 * max_iter)
    test_every_iter = int(test_iters[k] * ratio * muls[i])
    take = int(math.ceil(train_num[k] * ratio))

    cmds = [
        script,
        f'SOLVER.gpu {gpu},',
        'SOLVER.logdir logs/seg/{}/{}/ratio_{:.2f}'.format(alias, name, ratio),
        f'SOLVER.max_iter {max_iter}',
        f'SOLVER.step_size {step_size1},{step_size2}',
        f'SOLVER.test_every_iter {test_every_iter}',
        f'SOLVER.test_iter {test_num[k]}',
        f'SOLVER.ckpt {ckpt}',
        f'DATA.train.location {data}/{name}_train_level3.tfrecords',
        f'DATA.train.take {take}',
        f'DATA.test.location {data}/{name}_test_level3.tfrecords',
        f'MODEL.nout {seg_num[k]}',
        f'MODEL.factor {factor}',
        f'LOSS.num_class {seg_num[k]}',
    ]

    cmd = ' '.join(cmds)
    print('\n', cmd, '\n')
    os.system(cmd)

summary = [
    'names, ' + ', '.join(names),
    'train_num, ' + ', '.join([str(x) for x in train_num]),
    'test_num, ' + ', '.join([str(x) for x in test_num]),
]
for i in range(len(ratios)-1, len(ratios)-2, -1):
  ious = [None] * len(names)
  for j in range(len(names)):
    filename = 'logs/seg/{}/{}/ratio_{:.2f}/test_summaries.csv'.format(alias, names[j], ratios[i])
    with open(filename, newline='') as fid:
      reader = csv.reader(fid)
      for k, row in enumerate(reader):
        if k == 0: idx = row.index(' iou')
    ious[j] = row[idx]
  summary.append('Ratio:{:.2f}, '.format(ratios[i]) + ', '.join(ious))

with open(f'logs/seg/{alias}/summaries.csv', 'w') as fid:
  summ = '\n'.join(summary)
  fid.write(summ)
  print(summ)
