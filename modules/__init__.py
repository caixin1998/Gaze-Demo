from .resnet import resnet50, resnet18, resnetx50
from .botnet import botnet
from .resnet50_scratch_dag import resnet50_scratch_dag
from .discriminator import Discriminator
from .completion_net import CompletionNetwork,LocalDis,GlobalDis,PredictionEyeNetwork
from .fsanet import FSANet
from .iTrackerModel import iTrackerECModel
from .gaze_process import GazeProcess
# from .vtp import VTP
from .hrnet import hrnet18, hrnet32, hrnet48, hrnet64
from .hratp import hratp18, hratp32, hratp48, hratp64
from .hrmap import hrmap18, hrmap32, hrmap48
from .convxnet import convnext_base, convnext_large, convnext_xlarge
# from .hrnet_w import hrnet_w64

from .iTrackerModel import iTrackerECModel
from .eye_net import EyeNet
from .dt_ed import DTED

#from .fsanet_64 import FSANet
__all__ = [
    'resnet18',
    'resnet50',
    'botnet',
    'resnet50_scratch_dag',
    'Discriminator',
    'CompletionNetwork',
    'LocalDis',
    'GlobalDis',
    'FSANet',
    'PredictionEyeNetwork',
    'iTrackerECModel',
    'GazeProcess',
    'hrnet18', 
    'hrnet32',
    'hrnet48',
    'hrnet64',
    'hratp18', 
    'hratp32',
    'hratp48',
    'hrmap18', 
    'hrmap32',
    'hrmap48',
    'hratp64',
    'resnetx50',
    'convnext_base',
    'convnext_large',
    'convnext_xlarge',
    'iTrackerECModel',
    'EyeNet',
    'DTED',
]
