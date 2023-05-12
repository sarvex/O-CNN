import os
import sys
import tensorflow as tf
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--octree", type=str, required=False,
                    default='../../octree')
parser.add_argument("--cuda", type=str, required=False,
                    default='/usr/local/cuda-10.1')
parser.add_argument('--key64', type=str, required=False,
                    default='false')
parser.add_argument('--cc', type=str, required=False,
                    default='g++')

args = parser.parse_args()
OCTREE_DIR = args.octree
CUDA_DIR = args.cuda
CC = args.cc
KEY64 = '-DKEY64' if args.key64.lower() == 'true' else ''
LIBS_DIR = os.path.dirname(os.path.realpath(__file__))

TF_CFLAGS = " ".join(tf.sysconfig.get_compile_flags())
TF_LFLAGS = " ".join(tf.sysconfig.get_link_flags())

## g++-4.8
env_gcc = 'echo using g++ '
if CC == 'g++-4.8':
  cmds = [
    'sudo apt-get update',
    'sudo apt-get install gcc-4.8 --yes',
    'sudo apt-get install g++-4.8 --yes',]
  cmd = ' && '.join(cmds)
  print(cmd)
  os.system(cmd)
  env_gcc = 'echo using g++-4.8 && export CC=gcc-4.8 && export CXX=g++-4.8 '


## build octree
octree_ext = os.path.join(OCTREE_DIR, 'external/octree-ext')
if not os.path.exists(octree_ext):
  cmd = f'git clone --recursive https://github.com/wang-ps/octree-ext.git {octree_ext}'
  print(cmd)
  os.system(cmd)

octree_build = os.path.join(OCTREE_DIR, 'build')
if os.path.exists(octree_build):
  os.system(f'rm -r {octree_build}')

os.makedirs(octree_build)
os.chdir(octree_build)
abi = 'OFF' if '-D_GLIBCXX_USE_CXX11_ABI=1' in TF_CFLAGS else 'ON'
k64 = 'OFF' if KEY64 != '-DKEY64' else 'ON'
cmd = f'{env_gcc}&& cmake .. -DABI={abi} -DKEY64={k64} && make -j all'
print(cmd)
os.system(cmd)
os.system('./octree_test')  # run the test
os.chdir(LIBS_DIR)


## build ocnn-tf
lines = [
    f"TF_CFLAGS := {TF_CFLAGS}",
    f"TF_LFLAGS := {TF_LFLAGS}",
    f"OCT_CFLAGS := -I{OCTREE_DIR}/octree",
    f"OCT_LFLAGS := -L{OCTREE_DIR}/build -loctree_lib",
    "",
    "NVCC_FLAGS1 := -std=c++11 -O2 -c",
    f"NVCC_FLAGS2 := $(TF_CFLAGS) $(OCT_CFLAGS) -I {CUDA_DIR}/include -x cu -Xcompiler -fPIC -D GOOGLE_CUDA=1 -I /usr/local   -expt-relaxed-constexpr -DNDEBUG {KEY64}",
    "CC_FLAGS1 := -std=c++11 -O2",
    f"CC_FLAGS2 := $(TF_CFLAGS) $(OCT_CFLAGS) -I {CUDA_DIR}/include   -L {CUDA_DIR}/lib64 -D GOOGLE_CUDA=1 $(TF_LFLAGS) $(OCT_LFLAGS) -fPIC -lcudart {KEY64}",
    "",
    f"CC := {CC}",
    f"NVCC := {CUDA_DIR}/bin/nvcc",
    "",
]
cpu_objects = []
cpu_sources = []
gpu_objects = []
gpu_sources = []
all_headers = []
for filename in sorted(os.listdir(".")):
  if filename.endswith(".cu.cc") or filename.endswith(".cu"):
    targetname = f"{filename}.o"
    gpu_sources.append(filename)
    gpu_objects.append(os.path.join("object", targetname))
  elif filename.endswith(".cc") or filename.endswith(".cpp"):
    targetname = f"{filename}.o"
    cpu_sources.append(filename)
    cpu_objects.append(os.path.join("object", targetname))
  elif filename.endswith(".h"):
    all_headers.append(filename)

lines.extend((
    f'cpu_objects := {" ".join(cpu_objects)}',
    f'gpu_objects := {" ".join(gpu_objects)}',
    "",
    ".PHONY : all clean",
    "",
    "all : libocnn.so",
    "",
    "libocnn.so : $(cpu_objects) $(gpu_objects)",
    "\t$(CC) $(CC_FLAGS1) -shared -o libocnn.so object/*.o $(CC_FLAGS2)",
    "",
))
for i in range(len(cpu_objects)):
  dependency = b" ".join(
      subprocess.Popen(
          f"{CC} -std=c++11 -MM -MG {cpu_sources[i]}",
          stdin=subprocess.PIPE,
          stdout=subprocess.PIPE,
          shell=True,
      ).stdout.readlines()).decode("utf-8")
  headers = [h for h in all_headers if dependency.find(h) > 0]
  headers = " ".join(headers)
  lines.append(f"{cpu_objects[i]} : {headers} {cpu_sources[i]}")
  lines.extend((
      "\t$(CC) $(CC_FLAGS1) -c %s $(CC_FLAGS2) -o %s" %
      (cpu_sources[i], cpu_objects[i]),
      "",
  ))
for i in range(len(gpu_objects)):
  dependency = b" ".join(
      subprocess.Popen(
          f"{CC} -std=c++11 -MM -MG {gpu_sources[i]}",
          stdin=subprocess.PIPE,
          stdout=subprocess.PIPE,
          shell=True,
      ).stdout.readlines()).decode("utf-8")
  headers = [h for h in all_headers if dependency.find(h) > 0]
  headers = " ".join(headers)
  lines.append(f"{gpu_objects[i]} : {headers} {gpu_sources[i]}")
  lines.extend((
      "\t$(NVCC) $(NVCC_FLAGS1) %s $(NVCC_FLAGS2) -o %s" %
      (gpu_sources[i], gpu_objects[i]),
      "",
  ))
lines.append("clean :")
lines.extend(("\t rm %s" % os.path.join("object", "*.o"), ""))
lines = [line + "\n" for line in lines]
with open("Makefile", "w") as f:
  f.writelines(lines)

# make
if os.path.exists("object"):
  os.system("rm -r object")
os.mkdir("object")
os.system("make -j all")

# test
os.chdir(f"{LIBS_DIR}/../test")
# env variable
cmd = 'export OCTREE_KEY=' + ('64' if KEY64 == '-DKEY64' else '32')
os.system(f"{cmd} && python test_all.py")
