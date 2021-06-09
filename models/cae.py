from .base_model import BaseModel
from . import networks
import torch
from utils import utils
from models import init_net
import os


#CAE는 convolutional autoencoder로 convolutional layer로 구성된 autoencoder이다.
class CAE(BaseModel):
    """This class implements the Convolutional AutoEncoder for normal image generation
    CAE is processed in encoder and decoder that is composed CNN layers
    """

    @staticmethod
    def add_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        return parser

    def __init__(self, opt):
        """Initialize the CAE model"""
        BaseModel.__init__(self, opt)
        self.opt = opt
        img_size = (self.opt.channels, self.opt.img_size, self.opt.img_size)
        latent = self.opt.latent
        #encoder 정의
        self.encoder = init_net(networks.Encoder(latent).cuda(), gpu = opt.gpu, mode = opt.mode)
        # initialize encoder networks doing data parallel and init_weights
        #decoder 정의 
        self.decoder = init_net(networks.Decoder(img_size, latent).cuda(), gpu = opt.gpu, mode = opt.mode)
        # initialize decoder networks doing data parallel and init_weights
        self.networks = ['encoder', 'decoder']
        self.criterion = torch.nn.MSELoss()
        self.visual_names = ['generated_imgs']
        self.model_name = self.opt.model
        self.loss_name = ['loss']

        #train mode일 때 optimize를 정의 해준다. --> optimizer는 Adam 이다. 
        if self.opt.mode == 'train':# if mode is train, we have to set optimizer and requires grad is true
            self.optimizer_e = torch.optim.Adam(self.encoder.parameters(), lr=self.opt.lr,
                                                betas=(self.opt.beta1, self.opt.beta2))
            self.optimizer_d = torch.optim.Adam(self.decoder.parameters(), lr=self.opt.lr,
                                                betas=(self.opt.beta1, self.opt.beta2))
            self.optimizers.append(self.optimizer_e)
            self.optimizers.append(self.optimizer_d)
            self.set_requires_grad(self.decoder, self.encoder, requires_grad=True)


    #forward를 하게 되면 이미지를 encoder에 넣어서features를 뽑고,
    #뽑은 features를 decoder에 넣어서 image를 generate를 하게 됩니다. 
    def forward(self):
        features = self.encoder(self.real_imgs)
        self.generated_imgs = self.decoder(features)

    def backward(self):
        self.loss = self.criterion(10*self.real_imgs, 10*self.generated_imgs)
        self.loss.backward()

    def set_input(self, input):
        self.real_imgs = input['img'].to(self.device)

    def train(self):
        #train을 하게 되면 forward를 하게 됩니다. 
        self.forward()
        #optimizer를 zero_gradient로 바궈줍니다. 
        self.optimizer_d.zero_grad()
        self.optimizer_e.zero_grad()
        #backward를 시키게 됩니다. loss를 구하고, backward를 하면서 loss를 줄이는 쪽으로 학습을 시킵니다. 
        self.backward()
        self.optimizer_d.step()
        self.optimizer_e.step()

    def test(self):
        with torch.no_grad():
            self.forward()

    def save_images(self, data):
        images = data['img']
        paths = os.path.join(self.opt.save_dir, self.opt.object)
        paths = os.path.join(paths, "result")
        anomaly_img = utils.compare_images(images, paths, self.generated_imgs, data, threshold=self.opt.threshold)
        #anomaly_img = utils.compare_images(images, self.generated_imgs, threshold=self.opt.threshold)
        utils.save_images(anomaly_img, paths, data)




