import os
import pathlib
import datatime
import argparse
import numpy as np

def main(arg):
    log_folder = arg.log
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    log_folder = pathlib.Path(log_folder)
    log_file = str(log_folder / datatime.datatime.now().__format__('%Y-%m-%d %H:%M'))
    
    grid_size = 32
    img_size = arg.img_size
    if arg.multi_scale:
        max_size = int(np.ceil(img_size / 0.667 / grid_size) * grid_size)
        min_size = int(np.ceil(img_size / 1.5 / grid_size) * grid_size)
        
    train_hyp = arg.train_hyp
    