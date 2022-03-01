import configargparse
import os
import torch
import models
# import data
# import pytorch_lightning as pl

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self, filename=None):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.filename = filename
    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # parser = pl.Trainer.add_argparse_args(parser)
        
        # basic parameters
        parser.add_argument('-c', '--config', required=False, is_config_file=True, help='config file path', default = self.filename)
        parser.add_argument('--name', type=str, default='gaze_estimation', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--seed', type=int, default=None, help='random seed for experiments.')
        parser.add_argument('--checkpoints_dir', type=str, default='./logs', help='models are saved here')
        # model parameters

        
        #for model
        parser.add_argument('--model', type=str, default='faze', help='chooses which model to use. ')
        parser.add_argument('--netGaze', type=str, default='regressor', help='type for network')
        parser.add_argument('--backbone', type=str, default='resnet50', help='backbone for network')
        parser.add_argument('--ngf', type=int, default='128', help='network filters in the last conv layer')

        #for demo
        parser.add_argument('--cal', dest='cal', default=False, action='store_true')
        parser.add_argument('--do_collect',  type=bool, default=False)
        parser.add_argument('--do_finetune', type=bool, default=False)

        parser.add_argument('--cal_weights_path', type = str, default = 'weights/calibration')
        parser.add_argument('--k', type=int, default=9, help='point number for calibration (maml)')

        parser.add_argument('--cam_idx', type=int, default=[4,6], nargs = '+',help='cam_idx')
        parser.add_argument('--id', type=str, default="test",help='cam_idx')
        parser.add_argument('--pose_estimator', type=str, default='pnp', help='pnp or eos')
        parser.add_argument('--patch_type', type=str, default='faze', help='faze, face or eyes')

        parser.add_argument('--cal_weight_path',  type=str, default='weights/calibration')
        parser.add_argument('--display_patch', type=bool, default=True, help='display the face')
        parser.add_argument('--visualize_cal', type=bool, default=False, help='display the face')


        parser.add_argument('--camera_size', type=int, nargs = '+', default=[1920, 1080], help='display the face')
        self.initialized = True
        return parser



    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = configargparse.ArgumentParser(formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        opt.default_root_dir = expr_dir
        os.makedirs(expr_dir, exist_ok=True)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.name))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""

        opt = self.gather_options()
        self.print_options(opt)
        self.opt = opt
        return self.opt
