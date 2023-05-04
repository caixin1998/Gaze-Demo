
import torch
from .base_model import BaseModel
from .networks import GazeRes18
from .losses import GazeAngularLoss
import os,sys
import numpy as np
sys.path.append("../src")
from utils import polyfit,polymat4tensor

#linear calibration model
class PloyFitModel(BaseModel):
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
        parser.add_argument('--ckpt_path', type=str, default='weights/gazeres18_vipl538.ckpt', help='parameters path')

        # add order for polyfit,dtype is int
        parser.add_argument('--order', type=int, default=1, help='order for polyfit')
       
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
        self.netG  = GazeRes18().to(self.device)
        self.matrix = None
            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            # We also provide a GANLoss class "networks.GANLoss". self.criterionGAN = networks.GANLoss().to(self.device)
        self.criterionLoss = GazeAngularLoss(key_true = "gaze", key_pred="pred")
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
                # print(k, input[k].shape)
            # print(input[k].device)
        return input

    def forward(self, input):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        
        self.input = self.trans_totensor(input)
        # torchvision.utils.save_image(input['face'], "test.png")
        self.output = self.netG(self.input) 
        if self.matrix is not None:
            output = self.output["pred"]
            matrix = self.matrix
            self.output["pred"] = polymat4tensor(output, matrix, order=self.opt.order) 
        self.output["gaze"] = self.output["pred"]
        return self.output


        

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def optimize_parameters(self,input):
        """Update network weights; it will be called in every training iteration."""
        # self(input)               # first call forward to calculate intermediate results
        # self.optimizer.zero_grad()   # clear network G's existing gradients
        # self.backward()              # calculate gradients for network G
        # self.optimizer.step()
        # update gradients for network G
        for k, v in input.items():
            input[k] = v[:,...]
        with torch.no_grad():
            self(input)
        output = self.output["pred"]
        target = input["gaze"]
        self.matrix = polyfit(output.cpu().data.numpy(), target.cpu().data.numpy(),order=self.opt.order)
        self.matrix = torch.FloatTensor(self.matrix).to(self.device)


    def test(self, input):
        with torch.no_grad():
            self(input)
        self.loss_G = self.criterionLoss(self.input, self.output) 
        return self.loss_G
    
    def load_ckpt(self, ckpt_path):
        if os.path.isfile(ckpt_path):
            print("=> loading checkpoint '{}'".format(ckpt_path))
            checkpoint = torch.load(ckpt_path)
            for k, v in list(checkpoint["state_dict"].items()):
                if k.startswith("netGazeNetwork."):
                    checkpoint["state_dict"][k[15:]] = v
                    del checkpoint["state_dict"][k]
            self.netG.load_state_dict(checkpoint["state_dict"])

    def load_init_networks(self):
        assert os.path.isfile(self.opt.ckpt_path)
        self.load_ckpt(self.opt.ckpt_path)
        self.matrix = None

    def load_networks(self, subject):
        save_weight = torch.load('weights/calibration/%s_polyfit.pth.tar' % subject)
        self.netG.load_state_dict(save_weight["model"])
        self.matrix = save_weight["matrix"]

    def save_networks(self, subject):
        save_weight = {}
        state_dict = self.netG.state_dict()
        save_weight["model"] = state_dict
        save_weight["matrix"] = self.matrix
        torch.save(save_weight, 'weights/calibration/%s_polyfit.pth.tar' % subject) 

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


        input_dict_train = {
            'face': img[:3*k, :, :, :],
            'gaze': gaze_a[:3*k, :],
        }

        input_dict_valid = {
            'face': img[3*k:, :, :, :],
            'gaze': gaze_a[3*k:, :],
        }

        return input_dict_train, input_dict_valid

        
        # pass