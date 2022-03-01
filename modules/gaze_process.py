from .neural_process import NeuralProcess
import torch.nn as nn
import torch
from models import networks
from random import randint
class GazeProcess(nn.Module):
    def __init__(self, opt):
        super(GazeProcess, self).__init__()
        # gaze_backbone = list(networks.define_network(opt).children())
        # self.gaze_net = nn.Sequential(*gaze_backbone[:-1])
        self.process = NeuralProcess(opt)
        self.opt = opt
        self.upsample = torch.nn.Upsample(self.opt.y_dim)
        # self.num_context_range = (opt.support_size_lb, opt.support_size)
    def forward(self, input_dict, is_train = True):
        output_dict = {}
        for k, v in input_dict.items():
            if not isinstance(v, dict):
                input_dict[k] = v.view(v.shape[0], (v.shape[1] * v.shape[2]), *(v.shape[3:]))
            else:
                for k1, v1 in v.items():
                    v[k1] = v1.view(v1.shape[0], (v1.shape[1] * v1.shape[2]), *(v1.shape[3:]))
        support_features = input_dict["support_set"]["feature"]
        query_features = input_dict["query_set"]["feature"]

        supquery_features = torch.cat([support_features, query_features], dim=1)
        # print("input_dict[\"support_targets\"][1,1,...]",input_dict["support_targets"][1,1,...])

        support_targets = self.upsample(input_dict["support_targets"])
        query_targets = self.upsample(input_dict["query_targets"])
        # print(support_targets.shape, support_targets[1,1,...])
        # num_context = randint(self.num_context_range)
        supquery_targets = torch.cat([support_targets, query_targets], dim=1)
        # print("support_features, support_targets, supquery_features, supquery_targets",support_features.shape, support_targets.shape, supquery_features.shape, supquery_targets.shape)
        output_dict = self.process(support_features, support_targets, supquery_features, supquery_targets)
        output_dict["query_targets"] = query_targets
        output_dict["supquery_targets"] = supquery_targets

        output_dict["supquery_targets_pred"] = output_dict["y_pred_mu"]
        output_dict["supquery_targets_sigma"] = output_dict["y_pred_sigma"]
        output_dict["query_targets_pred"] = output_dict["supquery_targets_pred"][:,support_features.shape[1]:,...]
        output_dict["query_targets_sigma"] = output_dict["y_pred_sigma"][:,support_features.shape[1]:,...]

        return output_dict
