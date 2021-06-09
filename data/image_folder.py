import torch.utils.data as data
from PIL import Image
import os
from torchvision import transforms

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def is_imgfile(filename):
    """Checking if the file format is image format"""
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_datapaths(train_dir):
    """To create dataset, we need to save paths of images"""
    image_paths = []
    assert os.path.isdir(train_dir), f"{train_dir} is not existed"

    for root, _, names in os.walk(train_dir):
        for name in names:
            if is_imgfile(name):
                path = os.path.join(root, name)
                image_paths.append(path)
    return image_paths



def get_transform(opt):
    """Transforms images"""
    transform = []
    #처음에 이미지가 들어오게되면 rotation을 시켜라
    if opt.rotate == True:
        transform.append(transforms.RandomRotation(0.5))
    
    #밝기는 어느정도로 할 것이냐
    transform.append(transforms.ColorJitter(brightness=opt.brightness))
    #crop size를 가지고 이미지를 crop 시켜라  256x256
    transform.append(transforms.Resize((opt.cropsize, opt.cropsize), interpolation=2))
    transform.append(transforms.ToTensor())
    #option에 따라 channel을 몇개를 쓸 것이냐
    if opt.channels == 3:
        transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    elif opt.channels == 1:
        transform.append(transforms.Normalize((0.5), (0.5)))

    return transforms.Compose(transform)


class ImageFolder(data.Dataset):
    """ImageFolder functions for dataset"""

    def __init__(self, root, transform=None, return_paths=False):
        imgs = get_datapaths(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths


    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    
    def __len__(self):
        return len(self.imgs)

