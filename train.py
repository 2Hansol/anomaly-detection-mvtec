"""This code is for detecting anomaly parts in images
There are so many algorithms that help to detect anomaly like mvtecAD, anogan etc..
This scripts show anomaly detection using simple Auto Encoder.

you need to specify the folder which stores datasets(mvtec), and model you gonna use(AE, AAE)

Example:
    Train:
        python train.py --datadir [dataset folder] --model [model_name]
"""


from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import time
from utils.utils import plt_show




if __name__ == "__main__":
    #train을 하기 위해 먼저 train의 option을 가져옵니다. train의 option이란 train을 할 때 필요한 인자들을 말하게 됩니다. 
    opt = TrainOptions().parse()   
    #mvtec data를 가지고, dataset을 생성을 해야한다. 
    dataset = create_dataset(opt)  
    dataset_size = len(dataset)
    print(f"Training size is = {dataset_size}")

    #모델을 불러 옵니다. 
    model = create_model(opt) 
    
    # trian이면 스케쥴러를 정의하고, test이면, pretrained networks를 가지고 온다. 
    model.setup(opt)                
    total_iters = 0
    loss_name = model.loss_name            
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()                      
        #train이기 때문에 secheduler를 가져와서 learning_rate를 변경해준다. 
        model.update_learning_rate(epoch)  
        epoch_iters = 0


        #dataset에 대한 반복문이다. 
        for i, data in enumerate(dataset):                 
            iter_start_time = time.time()                  
            #batch size 만큼의 data를 model에 넣어 줍니다. 
            model.set_input(data)                          
            #train을 시킵니다. 
            model.train()                                 
            total_iters += 1
            epoch_iters += 1
        
        #model loss와 time에 대해 출력을 해줍니다. 
        if epoch % opt.print_epoch_freq == 0:               
            losses = model.get_current_losses(*loss_name)
            epoch_time = time.time() - epoch_start_time
            message = f"epoch : {epoch} | total_iters : {total_iters} | epoch_time:{epoch_time:.3f}"
            for k,v in losses.items():
                message += f" | {k}:{v}"
            print(message)
            
        #model을 저장해줍니다. 
        if epoch % opt.save_epoch_freq == 0:               
            print(
                "saving the latest model (epoch %d, total_iters %d)"
                % (epoch, total_iters)
            )
            model.save_networks()
            print('epoch : ',epoch, 'pth saved')
            #plt_show(model.generated_imgs[:3])
