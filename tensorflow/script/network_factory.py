import tensorflow as tf
from network_cls import network_ocnn, network_resnet
from network_unet import network_unet
from network_segnet import network_segnet
from network_hrnet import HRNet


def cls_network(octree, flags, training, reuse=False):
  if flags.name.lower() == 'ocnn':
    return network_ocnn(octree, flags, training, reuse)
  elif flags.name.lower() == 'resnet':
    return network_resnet(octree, flags, training, reuse)
  elif flags.name.lower() == 'hrnet':
    return HRNet(flags).network_cls(octree, training, reuse)
  else:
    print(f'Error, no network: {flags.name}')

def seg_network(octree, flags, training, reuse=False, pts=None, mask=None):
  if flags.name.lower() == 'unet':
    return network_unet(octree, flags, training, reuse)
  elif flags.name.lower() == 'segnet':
    return network_segnet(octree, flags, training, reuse)
  elif flags.name.lower() == 'hrnet':
    return HRNet(flags).network_seg(octree, training, reuse, pts, mask)
  else:
    print(f'Error, no network: {flags.name}')

    
