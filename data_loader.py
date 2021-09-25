# Standard libs
import os, re, shutil, sys, time, datetime, logging, pickle, json, socket
import random, math, gc, copy
from collections import OrderedDict
from ctypes import c_bool
from multiprocessing import Process, Value
from multiprocessing.managers import BaseManager
import multiprocessing, threading
import numpy as np
from collections import deque
from collections import OrderedDict
import collections
import numba

# PyTorch libs
import torch
from torch.multiprocessing import Process
from torch.multiprocessing import Queue
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as tormodels
from torch.utils.data.sampler import WeightedRandomSampler
from torch_baidu_ctc import CTCLoss

# libs from FLBench
from argParser import args
from utils.divide_data import partition_dataset, select_dataset, DataPartitioner
#from utils.models import *
from utils.utils_data import get_data_transform
from utils.utils_model import MySGD, test_model

if args.task == 'nlp':
    from utils.nlp import *
elif args.task == 'speech':
    from utils.speech import SPEECH
    from utils.transforms_wav import ChangeSpeedAndPitchAudio, ChangeAmplitude, FixAudioLength, ToMelSpectrogram, LoadAudio, ToTensor
    from utils.transforms_stft import ToSTFT, StretchAudioOnSTFT, TimeshiftAudioOnSTFT, FixSTFTDimension, ToMelSpectrogramFromSTFT, DeleteSTFT, AddBackgroundNoiseOnSTFT
    from utils.speech import BackgroundNoiseDataset

from helper.clientSampler import clientSampler
from utils.yogi import YoGi

# shared functions of aggregator and clients
# initiate for nlp
tokenizer = None

if args.task == 'nlp' or args.task == 'text_clf':
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True)

modelDir = os.path.join(args.log_path, args.model)
modelPath = modelDir+'/'+str(args.model)+'.pth.tar' if args.model_path is None else args.model_path

def init_dataset():
    global tokenizer

    outputClass = {'Mnist': 10, 'cifar10': 10, "imagenet": 1000, 'emnist': 47,
                    'openImg': 596, 'google_speech': 35, 'femnist': 62, 'yelp': 5
                }

    logging.info("====Initialize the model")

    if args.task == 'nlp':
        # we should train from scratch
        config = AutoConfig.from_pretrained(os.path.join(args.data_dir, 'albert-base-v2-config.json'))
        model = AutoModelWithLMHead.from_config(config)

    elif args.task == 'speech':
        if args.model == 'mobilenet':
            from utils.resnet_speech import mobilenet_v2
            model = mobilenet_v2(num_classes=outputClass[args.data_set], inchannels=1)
        elif args.model == "resnet18":
            from utils.resnet_speech import resnet18
            model = resnet18(num_classes=outputClass[args.data_set], in_channels=1)
        elif args.model == "resnet34":
            from utils.resnet_speech import resnet34
            model = resnet34(num_classes=outputClass[args.data_set], in_channels=1)
        elif args.model == "resnet50":
            from utils.resnet_speech import resnet50
            model = resnet50(num_classes=outputClass[args.data_set], in_channels=1)
        elif args.model == "resnet101":
            from utils.resnet_speech import resnet101
            model = resnet101(num_classes=outputClass[args.data_set], in_channels=1)
        elif args.model == "resnet152":
            from utils.resnet_speech import resnet152
            model = resnet152(num_classes=outputClass[args.data_set], in_channels=1)
        else:
            # Should not reach here
            logging.info('Model must be resnet or mobilenet')
            sys.exit(-1)
    else:
        model = tormodels.__dict__[args.model](num_classes=outputClass[args.data_set])

    if args.load_model:
        try:
            with open(modelPath, 'rb') as fin:
                model = pickle.load(fin)

            logging.info("====Load model successfully\n")
        except Exception as e:
            logging.info("====Error: Failed to load model due to {}\n".format(str(e)))
            sys.exit(-1)

    train_dataset, test_dataset = [], []

    # Load data if the machine acts as clients
    if args.this_rank != 0:

        if args.data_set == 'blog':
            train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
            test_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

        elif args.data_set == 'stackoverflow':
            from utils.stackoverflow import stackoverflow

            train_dataset = stackoverflow(args.data_dir, train=True)
            test_dataset = stackoverflow(args.data_dir, train=False)

        elif args.data_set == 'google_speech':
            bkg = '_background_noise_'
            data_aug_transform = transforms.Compose([ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(), TimeshiftAudioOnSTFT(), FixSTFTDimension()])
            bg_dataset = BackgroundNoiseDataset(os.path.join(args.data_dir, bkg), data_aug_transform)
            add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
            train_feature_transform = transforms.Compose([ToMelSpectrogramFromSTFT(n_mels=32), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
            train_dataset = SPEECH(args.data_dir, train= True,
                                    transform=transforms.Compose([LoadAudio(),
                                             data_aug_transform,
                                             add_bg_noise,
                                             train_feature_transform]))
            valid_feature_transform = transforms.Compose([ToMelSpectrogram(n_mels=32), ToTensor('mel_spectrogram', 'input')])
            test_dataset = SPEECH(args.data_dir, train=False,
                                    transform=transforms.Compose([LoadAudio(),
                                             FixAudioLength(),
                                             valid_feature_transform]))
        else:
            print('DataSet must be {}!'.format(['Mnist', 'Cifar', 'openImg', 'blog', 'stackoverflow', 'speech', 'yelp']))
            sys.exit(-1)

    return model, train_dataset, test_dataset
