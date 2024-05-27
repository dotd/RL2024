import argparse
import logging
import os
from datetime import time
from pathlib import Path
import os

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import deque
import tkinter


class Label:

    def __init__(self):
        # GUI for saving models with Tkinter
        self.FONT = "Fixedsys 12 bold"  # GUI font
        self.save_command1 = 0
        self.save_command2 = 0
        self.save_command3 = 0
        self.load_command_1 = 0
        self.load_command_2 = 0
        self.load_command_3 = 0
        self.resume_command = 0
        self.stop_command = 0
        self.window = tkinter.Tk()
        self.window.lift()
        self.window.attributes("-topmost", True)
        self.window.title("DQN-Vision Manager")
        self.lbl = tkinter.Label(self.window, text="Manage training -->")
        self.lbl.grid(column=0, row=0)
        self.btn1 = tkinter.Button(self.window, text="Save 1", font=self.FONT, command=self.clicked1, bg="gray")
        self.btn1.grid(column=1, row=0)
        self.btn2 = tkinter.Button(self.window, text="Save 2", font=self.FONT, command=self.clicked2, bg="gray")
        self.btn2.grid(column=2, row=0)
        self.btn3 = tkinter.Button(self.window, text="Save Best", font=self.FONT, command=self.clicked3, bg="gray")
        self.btn3.grid(column=3, row=0)
        self.load_btn1 = tkinter.Button(self.window, text="Load 1", font=self.FONT, command=self.clicked_load1,
                                        bg="blue")
        self.load_btn1.grid(column=1, row=1)
        self.load_btn2 = tkinter.Button(self.window, text="Load 2", font=self.FONT, command=self.clicked_load2,
                                        bg="blue")
        self.load_btn2.grid(column=2, row=1)
        self.load_btn3 = tkinter.Button(self.window, text="Load Best", font=self.FONT, command=self.clicked_load3,
                                        bg="blue")
        self.load_btn3.grid(column=3, row=1)
        self.resume_btn = tkinter.Button(self.window, text="Resume Training", font=self.FONT,
                                         command=self.clicked_resume, bg="green")
        self.resume_btn.grid(column=1, row=2)
        self.stop_btn = tkinter.Button(self.window, text="Stop Training", font=self.FONT, command=self.clicked_stop,
                                       bg="red")
        self.stop_btn.grid(column=3, row=2)

    def clicked1(self):
        self.lbl.configure(text="Model saved in slot 1!")
        self.save_command1 = True

    def clicked2(self):
        self.lbl.configure(text="Model saved in slot 2!")
        self.save_command2 = True

    def clicked3(self):
        self.lbl.configure(text="Model saved in slot 3!")
        self.save_command3 = True

    def clicked_load1(self):
        self.lbl.configure(text="Model loaded from slot 1!")
        self.load_command_1 = True

    def clicked_load2(self):
        self.lbl.configure(text="Model loaded from slot 2!")
        self.load_command_2 = True

    def clicked_load3(self):
        self.lbl.configure(text="Model loaded from slot 3!")
        self.load_command_3 = True

    def clicked_resume(self):
        self.lbl.configure(text="Training resumed!")
        self.resume_command = True

    def clicked_stop(self):
        self.lbl.configure(text="Training stopped!")
        self.stop_command = True


class Trainer:

    def __init__(self):
        pass


def run_main(args):
    graph_name = f'Cartpole_Vision_Stop-{args.TRAINING_STOP}_LastEpNum-{args.LAST_EPISODES_NUM}'
    save_graph_folder = 'save_graph/'
    os.makedirs(save_graph_folder) if not os.path.exists(save_graph_folder) else None


def prepare_parameters_and_logging():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="DEBUG", help="Set the logging level (default is DEBUG).")
    parser.add_argument("--BATCH_SIZE", type=int, default=128)
    parser.add_argument("--GAMMA", type=float, default=0.999)
    parser.add_argument("--EPS_START", type=float, default=0.9)
    parser.add_argument("--EPS_END", type=float, default=0.05)
    parser.add_argument("--EPS_DECAY", type=int, default=5000)
    parser.add_argument("--TARGET_UPDATE", type=int, default=50)
    parser.add_argument("--MEMORY_SIZE", type=int, default=100000)
    parser.add_argument("--END_SCORE", type=int, default=200)
    parser.add_argument("--USE_CUDA", type=bool, default=False)
    parser.add_argument("--LOAD_MODEL", type=bool, default=False)
    parser.add_argument("--GRAYSCALE", type=bool, default=True)
    parser.add_argument("--TRAINING_STOP", type=int, default=142)
    parser.add_argument("--N_EPISODES", type=int, default=50000)
    parser.add_argument("--LAST_EPISODES_NUM", type=int, default=20)
    parser.add_argument("--FRAMES", type=int, default=2)
    parser.add_argument("--RESIZE_PIXELS", type=int, default=60)
    parser.add_argument("--HIDDEN_LAYER_1", type=int, default=16)
    parser.add_argument("--HIDDEN_LAYER_2", type=int, default=32)
    parser.add_argument("--HIDDEN_LAYER_3", type=int, default=32)
    parser.add_argument("--KERNEL_SIZE", type=int, default=5)
    parser.add_argument("--STRIDE", type=int, default=2)
    parser.add_argument("--signature", type=str, default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--model_name", type=str, default='breakout_model')
    parser.add_argument("--model_save_interval", type=int, default=11)
    parser.add_argument("--logging_interval", type=int, default=4)
    parser.add_argument("--FONT", type=str, default="Fixedsys 12 bold")

    args = parser.parse_args()
    filename_full = os.path.abspath(__file__)
    filename = filename_full.split("/")[-1].split(".")[0]
    log_file = f"./logs/{filename}_{args.signature}.log"
    log_file_latest = f"./logs/{filename}.log"
    os.makedirs("./logs/") if not os.path.exists("./logs/") else None
    logging.basicConfig(level=getattr(logging, args.log_level), handlers=[logging.FileHandler(log_file),
                                                                          logging.FileHandler(log_file_latest),
                                                                          logging.StreamHandler()])
    # Put into logging the file itself for debugging
    logging.info(f"Running file: {filename_full}")
    logging.info(f"log_file: {log_file}")
    logging.info((f"Log file:\n{'start_of_running_file'.upper()}\n"
                  f"{Path(filename_full).read_text()}\n{'end_of_running_file'.upper()}"))
    for arg, value in vars(args).items():
        logging.info(f"{__file__.split('/')[-1]}> {arg}: {value}")
    return args


def main():
    args = prepare_parameters_and_logging()
    run_breakout(args)


if __name__ == '__main__':
    main()
    # cProfile.run('main()')  # Run the main function with cProfile
