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
from .networks import GazeNetwork
from .losses import GazeAngularLoss
import os,sys
import random
import numpy as np
import torchvision
sys.path.append("../src")
from utils import calculate_rotation_matrix
from tensor_utils import vector_to_pitchyaw
class FTModel(BaseModel):
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
        parser.add_argument('--ckpt_path', type=str, default='weights/eve_face.ckpt', help='parameters path')

        # parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for SGD')
        
        parser.add_argument('--atp_merge', type=str, default="add", help='method for merging for features in hratp.') 
        parser.add_argument('--patch_size', type=int, default=[56,28,14,7],nargs='+', help='the stirde of the first two 3x3 conv of hrnet48') 
        parser.add_argument('--hr_stride', type=int, default=[2,2],nargs='+', help='the stirde of the first two 3x3 conv of hrnet48') 
        parser.add_argument('--upsample_size', type=int, default=None, help='upsample size for input faces.') 
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
        self.criterionLoss = GazeAngularLoss(key_true = "gaze", key_pred="pred")
            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        self.optimizer = torch.optim.SGD(
                [p for n, p in self.netG.named_parameters() if n.startswith('gaze_fc')],
                lr=self.opt.lr,
            )
        
        for n, p in self.netG.named_parameters():
            if not n.startswith('gaze_fc'):
                p.requires_grad = False


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
        # torchvision.utils.save_image(input['face'], "test.png")
        
        self.output = self.netG(self.input)  
        self.output["gaze"] = self.output["pred"]
        return self.output

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_G = self.criterionLoss(self.input, self.output) 
        self.loss_G.backward()       # calculate gradients of network G w.r.t. loss_G

    def optimize_parameters(self,input):
        """Update network weights; it will be called in every training iteration."""
        # self(input)               # first call forward to calculate intermediate results
        # self.optimizer.zero_grad()   # clear network G's existing gradients
        # self.backward()              # calculate gradients for network G
        # self.optimizer.step()
        # update gradients for network G

        num_data = len(input["face"])
        train_list = list(range(num_data))
        random.shuffle(train_list)
        start = 0

        while start  < num_data:
            batch_input = {}
            if start + self.opt.batch_size > num_data:
                end =  num_data
            else: 
                end = start + self.opt.batch_size
            for k, v in input.items():
                batch_input[k] = v[train_list[start: end],...]

            self(batch_input)               # first call forward to calculate intermediate results
            self.optimizer.zero_grad()   # clear network G's existing gradients
            self.backward()              # calculate gradients for network G
            self.optimizer.step()

            start = end

    def test(self, input):
        with torch.no_grad():
            self(input)
        self.loss_G = self.criterionLoss(self.input, self.output) 
        return self.loss_G

    def load_init_networks(self):
        assert os.path.isfile(self.opt.ckpt_path)
        ckpt = torch.load(self.opt.ckpt_path)
        print("> Loading:",self.opt.ckpt_path)
        weights = dict([(k[:], v) for k, v in ckpt['state_dict'].items()])
        self.netG.load_state_dict(weights)
        
    def load_networks(self):
        load_path = os.path.join(self.opt.cal_weight_path, '%s'%(self.opt.subject),self.opt.ckpt_path.split('/')[-1])
        
        if os.path.isfile(load_path):
            ted_weights = torch.load(load_path)
            self.netG.load_state_dict(ted_weights)
            print("> Loading:",load_path)
        else:
            self.load_init_networks()

    def save_networks(self, subject):
        # pass
        save_path = os.path.join(self.opt.cal_weight_path, '%s'%subject)
        os.makedirs(save_path, exist_ok= True)
        torch.save(self.netG.state_dict(), os.path.join(save_path, self.opt.ckpt_path.split('/')[-1]))

    def convert_input(self, input):
        data = {
            'face': input['patch'], 
            # 'gt': input["normalized_gaze"]
        }
        return data

    def init_input(self, pre_data):
        
        k = self.opt.k
        data = {
            'face': pre_data['patch'], 
            'gaze' : pre_data['normalized_gaze'], 
        }

        n = len(data['face'])
        # print(n)
        #assert n==130, "Face not detected correctly. Collect calibration data again."
        _, c, h, w = data['face'][0].shape
        img = np.zeros((n, c, h, w))
        gaze_a = np.zeros((n, 2))

        for i in range(n):
            img[i, :, :, :] = data['face'][i]
            gaze_a[i, :] = data['gaze'][i]

        # create data subsets
        train_indices = []
        for i in range(0, k*10, 10):
            train_indices.append(random.sample(range(i, i + 10), 1))
        train_indices = sum(train_indices, [])

        valid_indices = []
        for i in range(k*10, n - 10, 10):
            valid_indices.append(random.sample(range(i, i + 10), 1))
        valid_indices = sum(valid_indices, [])
        input_dict_train = {
            'face': img[train_indices, :, :, :],
            'gaze': gaze_a[train_indices, :],
        }

        input_dict_valid = {
            'face': img[valid_indices, :, :, :],
            'gaze': gaze_a[valid_indices, :],
        }

        return input_dict_train, input_dict_valid

        
        # pass