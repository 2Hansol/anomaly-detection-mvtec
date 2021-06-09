from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    """This class is for defining arguments during only training."""

    def initialize(self, parser):
        #base option을 기본적으로 가지고 온다. 
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--print_epoch_freq', type=int, default=1, help='frequency of showing on screen')
        #epoch 몇번에 한번씩 저장을 해줄 것인지 
        parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of save point')
        #이미 pretrained가 되어 있을 경우에 추가로 하여서 학습을 이어가게끔 한다. 
        parser.add_argument('--epoch_count', type=int, default=3001, help='the starting point of epoch if you have pretrained model, you can start from that point')
        #epoch 수 ---> 몇번을 학습시킬 것인지
        parser.add_argument('--n_epochs', type=int, default=3050, help='number of epochs')
        
        #decay는 learning rate 스케쥴러 이다. train을 하면 할 수록 lr를 변화를 시켜주는 것에 대한 것.
        parser.add_argument('--n_epochs_decay', type=int, default=10, help='the number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of Adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of Adam')
        parser.add_argument('--lr', type=float, default=0.0002)
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        parser.add_argument('--mode', type=str, default='train', help='train or pretrained mode')
        
        # 기본적으로 cpu랑 gpu속도가 느리기 때문에 사용합니다. 
        parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
        
        #데이터 transformation에 관한 것 
        parser.add_argument('--no_dropout', action='store_true', help='no dropout')
        parser.add_argument('--rotate', action='store_false', help='rotate image in tranforms')
        parser.add_argument('--brightness', default=0.1, type=float, help='change brightness of images in tranforms')
        return parser
