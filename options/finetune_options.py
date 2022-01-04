from .base_options import BaseOptions


class FinetuneOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # for model
        parser.add_argument('--visual_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for SGD')
        parser.add_argument('--phase', type=str, default="train", help='train|test')
        parser.add_argument('--metric', type=str, default='angular', help='metrics')
        parser.add_argument('--criterion', type=str, default='smoothl1', help='models are saved here')

        #for dataloader
        parser.add_argument('--dataset', type=str, default='xgaze', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
        parser.add_argument('--num_threads', default=16, type=int, help='# threads for loading data')


        # for dataset
        parser.add_argument('--dataroot',  type=str, default= "/home/caixin/GazeData/xgaze_224", help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        
        parser.add_argument('--max_dataset_size', type=int, default=None, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--preprocess', type =str, default = "none", help='new dataset option')

        self.isTrain = True
        return parser