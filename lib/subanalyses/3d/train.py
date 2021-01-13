import argparse

from base import Config, train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--ld', default=0.3, type=float)
    parser.add_argument('--wd', default=0.001, type=float)
    parser.add_argument('--dr', default=1, type=int)

    parser.add_argument('--ar', default='peng', type=str)

    parser.add_argument('--ch', nargs='+', default=(32, 64, 128, 256, 256, 64), type=int)
    parser.add_argument('--ta', default='ageC', type=str)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)

    args = parser.parse_args()

    cfg = Config(learning_rate=args.lr, lr_decay=args.ld, weight_decay=args.wd, dropout=bool(args.dr), arch=args.ar,
                 channels=args.ch, target=args.ta, seed=args.seed, gpu=args.gpu)
    train(cfg)
