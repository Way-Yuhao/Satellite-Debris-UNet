import argparse


def main():
    parser = argparse.ArgumentParser(description='Specify target GPU, else the one defined in config.py will be used.')
    parser.add_argument('--gpu', type=int, help='cuda:$')
    args = parser.parse_args()
    if args.gpu is not None:
        CUDA_DEVICE = "cuda:{}".format(args.gpu)
    else:
        CUDA_DEVICE = "cpu".format(args.gpu)
    print(CUDA_DEVICE)


if __name__ == '__main__':
    main()
