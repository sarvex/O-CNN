import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, required=False, default='0202_linear')
parser.add_argument('--gpu', type=int, required=False, default=0)
parser.add_argument('--ckpt', type=str, required=False,
                    default='dataset/midnet_data/mid_d6_o6/model/iter_800000.ckpt')

args = parser.parse_args()
alias = args.alias
gpu = args.gpu
ckpt = args.ckpt

depth = 6
factor = 2
block_num = 3
decay = 5e-7
offset = 0.0
depth_out = 6

data = 'dataset/ModelNet40/m40_y'
ratios     = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
train_nums = [ 116,  214,  511, 1000, 1980, 4931, 9842]
muls       = [   4,    4,    2,    2,    1,    1,    1]

# run commands
def run_cmd(cmds):
  cmd = ' '.join(cmds)
  print('\n', cmd, '\n')
  os.system(cmd)

# testing feature
cmds = [
    'python  feature.py',
    f'SOLVER.gpu {gpu},',
    f'SOLVER.logdir logs/m40/{alias}/feature_cls',
    f'SOLVER.ckpt {ckpt}',
    'SOLVER.test_iter 2468',
    'SOLVER.run cls',
    'MODEL.channel 4',
    'MODEL.signal_abs True',
    'MODEL.nouts 128,64',
    f'MODEL.depth {depth}',
    f'MODEL.factor {factor}',
    f'MODEL.resblock_num {block_num}',
    f'MODEL.depth_out {depth_out}',
    f'DATA.test.location {data}_1.00_test_points.tfrecords',
    'DATA.test.batch_size 1',
    'DATA.test.distort False',
    'DATA.test.shuffle 0',
    'DATA.test.node_dis True',
    f'DATA.test.depth {depth}',
    f'DATA.test.offset {offset}',
]
run_cmd(cmds)


for i in range(len(ratios)):
  # training feature
  ratio = ratios[i]
  cmds = [
      'python  feature.py',
      f'SOLVER.gpu {gpu},',
      f'SOLVER.logdir logs/m40/{alias}/feature_cls',
      f'SOLVER.ckpt {ckpt}',
      f'SOLVER.test_iter {train_nums[i]}',
      'SOLVER.run cls',
      'MODEL.channel 4',
      'MODEL.signal_abs True',
      'MODEL.nouts 128,64',
      f'MODEL.depth {depth}',
      f'MODEL.factor {factor}',
      f'MODEL.resblock_num {block_num}',
      f'MODEL.depth_out {depth_out}',
      'DATA.test.location {}_{:.2f}_train_points.tfrecords'.format(
          data, ratio),
      'DATA.test.batch_size 1',
      'DATA.test.distort False',
      'DATA.test.scale 0.25',
      'DATA.test.jitter 0.125',
      'DATA.test.axis y',
      'DATA.test.angle 1,1,1',
      'DATA.test.uniform True',
      'DATA.test.shuffle 0',
      'DATA.test.node_dis True',
      f'DATA.test.depth {depth}',
      f'DATA.test.offset {offset}',
  ]
  run_cmd(cmds)

  # train the linear classifier
  step_size1 = int(200000 * ratio * muls[i])
  step_size2 = int(100000 * ratio * muls[i])
  max_iter = int(300000 * ratio * muls[i])
  prefix = f'logs/m40/{alias}/feature_cls/m40_y'
  train_data = '{}_{:.2f}_train_points.tfrecords'.format(prefix, ratio)
  test_data = f'{prefix}_1.00_test_points.tfrecords'
  cmds = [
      'python run_linear_cls.py',
      f'SOLVER.gpu {gpu},',
      'SOLVER.logdir logs/m40/{}/cls_fc1_{:.2f}'.format(alias, ratio),
      'SOLVER.run train',
      f'SOLVER.max_iter {max_iter}',
      'SOLVER.learning_rate 1.0',
      'SOLVER.test_iter 617',
      'SOLVER.test_every_iter 1000',
      f'SOLVER.step_size {step_size1},{step_size2}',
      f'DATA.train.location {train_data}',
      'DATA.train.batch_size 32',
      'DATA.train.x_alias fc1',
      'DATA.train.y_alias label',
      f'DATA.test.location {test_data}',
      'DATA.test.shuffle 0',
      'DATA.test.batch_size 4',
      'DATA.test.x_alias fc1',
      'DATA.test.y_alias label',
      'MODEL.name linear',
      'MODEL.nout 40',
      'LOSS.num_class 40',
      f'LOSS.weight_decay {decay}',
  ]
  run_cmd(cmds)

