# import taichi as ti
import time
import argparse
import torch

# ti.init(arch=ti.gpu)

# aa = ti.field(dtype=float, shape=1 << 27)
# bb = ti.field(dtype=float, shape=1 << 27)
# result = ti.field(dtype=float, shape=1 << 27)


# @ti.kernel
# def ComputeOnMetal():
#     for i in range(1 << 27):
#         aa[i] = ti.random()
#         bb[i] = ti.random()

#     print(time.time() - t0)
#     for i in range(1 << 27):
#         result[i] = aa[i] * bb[i]

#     for i in range(1 << 27):
#         result[i] = aa[i] / bb[i]
#     # result = aa + bb

#     # time.sleep(3)
#     print((time.time() - t0))


def ComputeOnMPS():
    aa = torch.randn(1 << 27, device='mps')
    bb = torch.randn(1 << 27, device='mps')

    start = time.time()
    result = aa * bb
    result = aa / bb

    print(time.time() - start)


def ComputeOnCPU():
    aa = torch.randn(1 << 27, device='cpu')
    bb = torch.randn(1 << 27, device='cpu')

    start = time.time()
    result = aa * bb
    result = aa / bb

    print(time.time() - start)


def ComputeOnCuda():
    aa = torch.randn(1 << 27).cuda()
    bb = torch.randn(1 << 27).cuda()

    # t0 = time.time()
    print(time.time() - t0)

    result = aa * bb
    result = aa / bb

    print(time.time() - t0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default='metal',
                        help='available parameter is [metal, cuda]')
    args = parser.parse_args()

    t0 = time.time()
    if args.gpu == 'metal':
        # ComputeOnMetal()
        pass

    elif args.gpu == 'cuda':
        ComputeOnCuda()

    elif args.gpu == 'cpu':
        ComputeOnCPU()

    else:
        ComputeOnMPS()
