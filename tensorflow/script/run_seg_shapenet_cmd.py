import os
import csv
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, required=False,
                    default='0204_shapenet_randinit')
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
module = 'run_seg_shapenet_finetune.py' if mode != 'randinit' else 'run_seg_shapenet.py'
script = f'python {module} --config configs/seg_hrnet_shapenet_pts.yaml'
if mode != 'randinit':
  script += f' SOLVER.mode {mode} '
data = 'dataset/shapenet_segmentation/datasets'


categories= ['02691156', '02773838', '02954340', '02958343',
             '03001627', '03261776', '03467517', '03624134',
             '03636649', '03642806', '03790512', '03797390',
             '03948459', '04099429', '04225987', '04379243']
names     = ['Aero',     'Bag',      'Cap',      'Car',
             'Chair',    'EarPhone', 'Guitar',   'Knife',
             'Lamp',     'Laptop',   'Motor',    'Mug',
             'Pistol',   'Rocket',   'Skate',    'Table']
seg_num   = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
train_num = [2348, 62, 44, 717, 3052, 55, 626, 312,
             1261, 367, 151, 146, 234, 54, 121, 4421]
test_num =  [341, 14, 11, 153, 693, 14, 159, 80,
             285, 78, 51, 38, 41, 12, 31, 842]
max_iters = [20000, 3000, 3000, 10000, 20000, 3000, 10000, 5000,
             10000, 5000, 5000,  5000,  5000, 3000,  5000, 20000]
#           [22012,  581,  412,  6722, 28612,  516,  5869, 2925, 
#            11822, 3441, 1416,  1369,  2194,  506,  1134, 41447] 
test_iters= [800, 100, 100, 400, 800, 100, 400, 200,
             400, 200, 200, 200, 200, 100, 200, 800]
ratios = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
# longer iterations when data < 10%
muls   = [   2,    2,    2,    1,    1,    1,    1]


for i in range(len(ratios)):
  for k in range(len(categories)):
    ratio, cat = ratios[i], categories[k]
    max_iter = int(max_iters[k] * ratio * muls[i])
    step_size1, step_size2 = int(0.5 * max_iter), int(0.25 * max_iter)
    test_every_iter = int(test_iters[k] * ratio * muls[i])
    take = int(math.ceil(train_num[k] * ratio))

    cmds = [
        script,
        f'SOLVER.gpu {gpu},',
        'SOLVER.logdir logs/seg/{}/{}_{}/ratio_{:.2f}'.format(
            alias, cat, names[k], ratio),
        f'SOLVER.max_iter {max_iter}',
        f'SOLVER.step_size {step_size1},{step_size2}',
        f'SOLVER.test_every_iter {test_every_iter}',
        f'SOLVER.test_iter {test_num[k]}',
        f'SOLVER.ckpt {ckpt}',
        f'DATA.train.location {data}/{cat}_train_val.tfrecords',
        f'DATA.train.take {take}',
        f'DATA.test.location {data}/{cat}_test.tfrecords',
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
for i in range(len(ratios)-1, -1, -1):
  ious = [None] * len(categories)  
  for j in range(len(categories)):
    name, cat, ratio = names[j], categories[j], ratios[i]
    filename = 'logs/seg/{}/{}_{}/ratio_{:.2f}/test_summaries.csv'.format(alias, cat, name, ratio)
    with open(filename, newline='') as fid:
      reader = csv.reader(fid)
      for k, row in enumerate(reader):
        if k == 0: idx = row.index(' iou')
    ious[j] = row[idx]
  summary.append('Ratio:{:.2f}, '.format(ratios[i]) + ', '.join(ious))

with open(f'logs/seg/{alias}/summaries.csv', 'w') as fid:
  summ = '\n'.join(summary)
  fid.write(summ)
  # print(summ)