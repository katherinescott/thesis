import argparse
import shlex
from itertools import product
from multiprocessing import Pool
from subprocess import check_output

N_JOBS = 4
DEFAULT_ARGS = {
    "context_size":  5,
    "epochs":  20,
    "batch_size": 128,
    "GPU": 0,
    "mode": "test",
    "l2": 1e-5,
    "optimizer":  "Adam",
    "init_weights": "rand_norm",
    "resume": "NONE",
    "start_epoch":  0,
    "model_dir": "models",
}


def runline_generator(kwarg_dict):
    kwarg_dict.update(DEFAULT_ARGS)
    argstr = ' '.join('--{} {}'.format(k, v) for k, v in kwarg_dict.items())
    return 'python main.py {}'.format(argstr)


def parse_runlines(args):
    arg_dict = vars(args)
    arg_lst = product(*arg_dict.values())
    return list(map(lambda x: runline_generator(
            dict(zip(arg_dict.keys(), x))), arg_lst))


def evaluate(runline):
    print('Running: {}'.format(runline))
    return runline, check_output(shlex.split(runline))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', nargs='+', default=0.001)
    parser.add_argument('--adapt_lr_epoch', nargs='+', default=5)
    parser.add_argument('--initial_lr', nargs='+', default=0.01)
    parser.add_argument('--dropout', nargs='+', default=0.0)
    args = parser.parse_args()
    runlines = parse_runlines(args=args)
    pool = Pool(processes=N_JOBS)
    min_perp = 1e5
    for res in pool.imap_unordered(
            evaluate, runlines, chunksize=len(runlines)//N_JOBS):
        if float(res[1]) < min_perp:
            min_perp = float(res[1])
            print('New best configuration found!\n{}'.format(res[0]))
    pool.close()
    pool.join()
