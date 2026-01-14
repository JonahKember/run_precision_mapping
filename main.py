import os
import json
import numpy as np
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--config')
args = parser.parse_args()

with open(str(args.config)) as f:
    cfg = json.load(f)


if __name__ == '__main__':

    output   = cfg['output']
    subjects = np.loadtxt(f"{cfg['subjects']}", dtype=str)

    for subject in subjects:

        if os.path.exists(f'{output}/data/sub-{subject}/networks.R.label.gii'):
            continue

        utils.clean_fmriprep_output(cfg, subject)
        utils.run_precision_mapping(cfg, subject)
