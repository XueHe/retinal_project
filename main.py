import argparse
import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random

def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s'%config.model_type)
        return
    
    # Create directories if not exist
    model_path=config.model_path
    result_path=config.result_path
    pre_type=config.pre_type
    model_type=config.model_type
    loss_type=config.loss_type
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path=config.model_path+pre_type+"_"+model_type+"_"+loss_type+"/"
    config.model_path=model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path=config.result_path+pre_type+"_"+model_type+"_"+loss_type+"/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    config.result_path=result_path
    
 
    

    lr = random.random()*0.0005 + 0.0000005
    augmentation_prob= random.random()*0.7
    epoch = config.num_epochs
    decay_ratio = random.random()*0.8
    decay_epoch = int(epoch*decay_ratio)

    config.augmentation_prob = augmentation_prob
    config.num_epochs = epoch
    config.lr = lr
    config.num_epochs_decay = decay_epoch

    print(config)
    data_path=config.data_path
    train_path=train_path=data_path+pre_type+"image_train/"
    valid_path=data_path+pre_type+"image_valid/"
    test_path=data_path+pre_type+"image_test/"
        
    train_loader = get_loader(image_path=train_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='train',
                            augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(image_path=valid_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='valid',
                            augmentation_prob=0.)
    test_loader = get_loader(image_path=test_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='test',
                            augmentation_prob=0.)

    solver = Solver(config, train_loader, valid_loader, test_loader)

    
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        best_model_path=config.best_model_path
        solver.test(best_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=600)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--pre_type', type=str, default='pre')
    parser.add_argument('--model_path', type=str, default='./models/')
    parser.add_argument('--data_path', type=str, default='./dataset/')
    parser.add_argument('--result_path', type=str, default='./result/')
    parser.add_argument('--cuda_idx', type=int, default=1)
    
    parser.add_argument('--loss_type',type=str, default='Dice')
    parser.add_argument('--best_model_path',type=str, default='best_model.pth')
    config = parser.parse_args()
    main(config)

    


    
  