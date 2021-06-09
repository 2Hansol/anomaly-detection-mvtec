import argparse
import torch
import os
from utils import utils


#base option으로 train, test 둘 다 에서 사용하게 된다. 
class BaseOptions():
    def initialize(self, parser):
        """Defining arguments used in both training and testing"""

        #data의 directory가 어디인지
        parser.add_argument('--data_dir', type=str, default = '/content/gdrive/MyDrive/anomaly-detection-mvtec/', help='path to dataset')
        #gpu를 몇번을 사용할 것 인지
        parser.add_argument('--gpu', type=str, default='0', help='gpu number : e.g. 0  0,1,2, 0,2. use -1 for CPU')
        #결과에 대한 파일은 어디에 저장할 것 인지
        parser.add_argument('--save_dir', type=str, default='/content/gdrive/MyDrive/anomaly-detection-mvtec/', help='models are saved here')
        #model은 어떤 model을 사용할 것 인지 --> Convolutional Auto Encoder 를 사용한다. 
        parser.add_argument('--model', type=str, default='cae', help='choose which model to use')
        #이미지의 채널은 몇 채널을 사용할 것인지 --> 보통 rgb로 3개의 channel이다. 
        parser.add_argument('--channels', type=int, default=3, help='# of image channels 3:RGB, 1:gray-scale')
        #이미지의 사이즈는 어떻게 지정할 것 인지
        parser.add_argument('--img_size', type=int, default=256, help='img size of input and output for networks')
        #이미지에서 features를 몇개를 뽑을 것인지.
        parser.add_argument('--latent', type=int, default=100, help='the letent vector size for networks')

        #처음 weigth을 initialize할 때 어떤식으로 할 것인지
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        
        #사용할 데이터에 관한 것
        parser.add_argument('--dataset', type=str, default='mvtec', help='chooses how datasets are loaded.')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--cropsize', type=int, default=256, help='crop image size')
        parser.add_argument('--object', type=str, default='capsule', help='the object for training')
        return parser

    def parse(self):
        """Parse base options and call function printing option .
           If # gpu is more than 0, make gpu ids list"""
        parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)

        opt = parser.parse_args()

        self.print_options(opt)

        gpu_ids = opt.gpu.split(',')
        opt.gpu = []
        for id in gpu_ids:
            id = int(id)
            if id >= 0:
                opt.gpu.append(id)
        if len(opt.gpu) > 0:
            torch.cuda.set_device(opt.gpu[0])

        return opt

    def print_options(self, opt):
        """print options and open save folder for saving options
           It will be saved in save_dir/model_name/[mode]opt.txt"""
        message = '----------------------Arguments-------------------------\n'
        for k, v in vars(opt).items():
            message += f'{k:>25}: {v:<30}\n'
        message += '---------------------End--------------------------------\n'
        print(message)

        # saving options
        result_dir = os.path.join(opt.save_dir, opt.model)
        utils.mkdirs(result_dir)
        opt_file_name = os.path.join(result_dir, f'{opt.mode}opt.txt')
        with open(opt_file_name, 'wt') as f:
            f.write(message)
