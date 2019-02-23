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
import util.util as util
import os

from options.train_options import TrainOptions

'''
python export_dataset.py --model_to_load saved_models/200_net_G_B.pth --dataroot datasets_demo/256/clean_relief_3k/ --model cycle_gan --which_model_netG unet_256 --input_nc 1 --output_nc 1
'''

class ExportOptions(TrainOptions):
    def initialize(self):
        TrainOptions.initialize(self)
        #self.parser.add_argument('--GA_model_to_load', type=str, default='checkpoints/terrain_cyclegan_clean_relief/200_net_G_A.pth', help='')
        self.parser.add_argument('--model_to_load', type=str, default='saved_models/200_net_G_B.pth', help='')
        self.parser.add_argument('--output-dir', type=str, default='output/', help='')


def remove_data_parallel(saved_model):
  from collections import OrderedDict
  filtered_dict = OrderedDict()
  for k, v in saved_model.items():
    k = k.replace('module.', '')
    filtered_dict[k] = v
  return filtered_dict


opt = ExportOptions().parse() 

# create directory if it doesn't exist
if not os.path.exists(opt.output_dir):
    os.makedirs(opt.output_dir)

def set_input(input, opt):
    input_A = input['A']
    input_B = input['B']
    input_A=input_A[:,0:opt.input_nc,:,:]
    input_B=input_B[:,0:opt.input_nc,:,:]
    image_paths = input['A_paths']
    real_A = Variable(input_A)
    real_B = Variable(input_B)
    if len(opt.gpu_ids) > 0:
        real_A = real_A.cuda(device=opt.gpu_ids[0])
        real_B = real_B.cuda(device=opt.gpu_ids[0])
    return real_A, real_B, image_paths


data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
#print('loading %s' % opt.GA_model_to_load)
#saved_model_A = torch.load(opt.GA_model_to_load, map_location=lambda storage, loc: storage)
print('loading %s' % opt.model_to_load)
saved_model_B = torch.load(opt.model_to_load, map_location=lambda storage, loc: storage)

#saved_model_A = remove_data_parallel(saved_model_A)
saved_model_B = remove_data_parallel(saved_model_B)


#model.netG_A.load_state_dict(saved_model_A)
model.netG_B.load_state_dict(saved_model_B)
#model.netG_A = model.netG_A.cuda(device=opt.gpu_ids[0])
model.netG_B = model.netG_B.cuda(device=opt.gpu_ids[0])
for i, data in enumerate(dataset):
    real_A, real_B, image_paths = set_input(data, opt)
    #fake_B = model.netG_A.forward(real_A)
    fake_A = model.netG_B.forward(real_B)
    real_B_img = util.tensor2im(real_B.data)
    fake_A_img = util.tensor2im(fake_A.data)
    util.save_image(real_B_img, opt.output_dir + "input_{:4d}_heightmap_8bit_normalized.png".format(i))
    util.save_image(fake_A_img, opt.output_dir + "target_{:4d}_reliefmap_8bit.png".format(i))
