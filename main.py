import os
import yaml
import pickle
from functools import partial
import argparse

from scaling_laws import scaling_law, fit_scaling_law, scaling_scatter

def main(args):
    with open(os.path.join(args.data_dir, 'config.yml')) as fle:
        cfg = yaml.safe_load(fle)

    with open(os.path.join(args.data_dir, 'runs.pkl'), 'rb') as fle:
        runs = pickle.load(fle)

    if 'fit' in cfg:
        params = fit_scaling_law(runs, grid_search = cfg['fit']['grid_search'])
        fit_fn = partial(scaling_law, params=params)
    else:
        fit_fn = None

    for chart in cfg['charts']:
        chart['savepath'] = os.path.join(args.data_dir, chart.pop('name') + '.html')
        scaling_scatter(runs=runs, **chart, fit_fn=fit_fn)


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str)

    args = parser.parse_args()
    main(args)


