import time
import torch
import torch.nn as nn
import torch.onnx
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

from options.train_options import TrainOptions

class ExportOptions(TrainOptions):
    def initialize(self):
        TrainOptions.initialize(self)
        self.parser.add_argument('--GA_model_to_load', type=str, default='checkpoints/terrain_cyclegan_clean_relief/200_net_G_A.pth', help='')
        self.parser.add_argument('--GB_model_to_load', type=str, default='checkpoints/terrain_cyclegan_clean_relief/200_net_G_B.pth', help='')

def export_onnx(the_model, input, filename, verbose=True):
  torch.onnx.export(the_model, input, filename, verbose=verbose, export_params=True)
  print("Successfully saved model to {}".format(filename))

def remove_data_parallel(saved_model):
  from collections import OrderedDict
  filtered_dict = OrderedDict()
  for k, v in saved_model.items():
    k = k.replace('module.', '')
    filtered_dict[k] = v
  return filtered_dict

opt = ExportOptions().parse() 
# for onnx export you have to put the model on the CPU
opt.gpu_ids = ''


data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
saved_model_A = torch.load(opt.GA_model_to_load, map_location=lambda storage, loc: storage)
saved_model_B = torch.load(opt.GB_model_to_load, map_location=lambda storage, loc: storage)

saved_model_A = remove_data_parallel(saved_model_A)
saved_model_B = remove_data_parallel(saved_model_B)


model.netG_A.load_state_dict(saved_model_A)
model.netG_B.load_state_dict(saved_model_B)

for i, data in enumerate(dataset):
    model.set_input(data)
    model.forward()
    model.netG_A.forward(model.real_A)
    model.netG_B.forward(model.real_B)
    input_shape = (1,3,256,256)
    export_onnx(model.netG_A, model.real_A,"saved_model_netG_A.onnx")
    export_onnx(model.netG_B, model.real_B,"saved_model_netG_B.onnx")
    exit()
