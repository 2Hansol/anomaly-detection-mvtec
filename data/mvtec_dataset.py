from data.base_dataset import BaseDataset
from data.image_folder import get_datapaths, get_transform
from PIL import Image
import os


class MvtecDataset(BaseDataset):


    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.img_size = opt.img_size
        self.train_dir = os.path.join(opt.data_dir, opt.object, opt.mode)
        self.train_paths = get_datapaths(self.train_dir)
        self.train_size = len(self.train_paths)
        self.transform = get_transform(opt)


    #dataset 경로에 있는 이미지들을 불러오기 위해서 getiten함수를 사용하여 image를 불러온다. 
    def __getitem__(self, index):
        img_path = self.train_paths[index % self.train_size]
        label = img_path.split('/')[-2]
        img = Image.open(img_path).convert('RGB')
        #가져온 이미지를 transform을 한다. 
        img = self.transform(img)

        return {'label' : label,'img' : img, 'path' : img_path}
    def __len__(self):
        return self.train_size