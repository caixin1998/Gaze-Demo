"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <trans_totensor>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
from .base_model import BaseModel
from modules import GazeNetwork
from .losses import GazeAngularLoss
import os,sys
import random
import numpy as np
import torchvision
sys.path.append("../src")
from utils import calculate_rotation_matrix
from tensor_utils import vector_to_pitchyaw
class EVEModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(dataset_mode='aligned')  # You can rewrite default values for this model. For example, this model usually uses aligned dataset as its dataset.
        parser.add_argument('--parameters_path', type=str, default='weights/eve_face.ckpt', help='parameters path')

        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for SGD')
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        # self.loss_names = ['loss_G']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        # self.visual_names = ['data_A', 'data_B', 'output']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        self.model_names = ['G']
        self.opt = opt
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.netG  = GazeNetwork(opt).to(self.device)

            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            # We also provide a GANLoss class "networks.GANLoss". self.criterionGAN = networks.GANLoss().to(self.device)
        self.criterionLoss = GazeAngularLoss()
            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        self.optimizer = torch.optim.SGD(
                [p for n, p in self.netG.named_parameters() if n.startswith('gaze')],
                lr=self.opt.lr,
            )

        # self.optimizers = [self.optimizer]

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def trans_totensor(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        for k, v in input.items():
            if isinstance(v, np.ndarray):
                input[k] = torch.FloatTensor(v).to(self.device).detach()
            # print(input[k].device)
        return input

    def forward(self, input):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        
        self.input = self.trans_totensor(input)
        torchvision.utils.save_image(input['face'], "test.png")
        self.output = self.netG(self.input)  
        return self.output

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_G = self.criterionLoss(self.input, self.output) 
        self.loss_G.backward()       # calculate gradients of network G w.r.t. loss_G

    def optimize_parameters(self,input):
        """Update network weights; it will be called in every training iteration."""
        self(input)               # first call forward to calculate intermediate results
        self.optimizer.zero_grad()   # clear network G's existing gradients
        self.backward()              # calculate gradients for network G
        self.optimizer.step()        # update gradients for network G

    def test(self, input):
        self(input)
        self.loss_G = self.criterionLoss(self.input, self.output) 
        return self.loss_G

    def load_init_networks(self):
        assert os.path.isfile(self.opt.parameters_path)
        ckpt = torch.load(self.opt.parameters_path)
        weights = dict([(k[15:], v) for k, v in ckpt['state_dict'].items()])
        self.netG.load_state_dict(weights)

    def load_networks(self):
        self.load_init_networks()

    def save_networks(self, subject):
        torch.save(self.netG.state_dict(), 'weights/calibration/%s_faze.pth.tar' % subject) 

    def convert_input(self, input):
        data = {
            'face': input['patch'], 
        }
        return data

    def init_input(self, pre_data):
        pass