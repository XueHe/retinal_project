import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
import torch.nn.functional as F
from evaluation import *
from network import U_Net, R2U_Net, AttU_Net, R2AttU_Net
import csv
import json
from pytorch_toolbelt import losses as L


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.pre_type=config.pre_type

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.augmentation_prob = config.augmentation_prob
        self.loss_type = config.loss_type

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode
        self.device = torch.device('cuda:' + str(config.cuda_idx) if torch.cuda.is_available() else 'cpu')
        print(self.device)

        # Early stopping parameters
        self.patience = 600
        self.best_val_loss = float('inf')
        self.early_stop_count = 0

        self.model_type = config.model_type
        self.t = config.t
        self.build_model()
        

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type == 'U_Net':
            self.unet = U_Net(img_ch=1, output_ch=1)
        elif self.model_type == 'R2U_Net':
            self.unet = R2U_Net(img_ch=1, output_ch=1, t=self.t)
        elif self.model_type == 'AttU_Net':
            self.unet = AttU_Net(img_ch=1, output_ch=1)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=1, output_ch=1, t=self.t)
        self.optimizer = optim.Adam(list(self.unet.parameters()), self.lr, [self.beta1, self.beta2])
        self.unet.to(self.device)
        self.checkpoint_file = "./models/"+self.pre_type+"/"+"model_type"+"/checkpoints/"

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = sum(p.numel() for p in model.parameters())
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        return x.cpu().data if torch.cuda.is_available() else x.data

    def update_lr(self, g_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = g_lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def val(self, epoch):
        self.unet.to(self.device)
        self.unet.train(False)
        self.unet.eval()

        # 保存模型checkpoint的路径
        checkpoint_dir = os.path.join(self.model_path, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        acc = 0.  # Accuracy
        SE = 0.  # Sensitivity (Recall)
        SP = 0.  # Specificity
        val_loss = 0
        length = 0

        if self.loss_type=="Dice":
                loss_function = L.DiceLoss(mode='binary')
        if self.loss_type=="BCE":
            loss_function = torch.nn.BCELoss()
        if self.loss_type == "mixed":
            bce_loss = torch.nn.BCELoss()
            dice_loss = L.DiceLoss(mode='binary')
            loss_function = lambda inputs, targets: bce_loss(inputs, targets) + dice_loss(inputs, targets)

        for i, (images, GT) in enumerate(self.valid_loader):
            images = images.to(self.device)
            GT = GT.to(self.device)
            SR = torch.sigmoid(self.unet(images))
            loss = loss_function(SR, GT)
            val_loss += loss.item()

            # 调用 calculate_each_image 函数，并将返回结果保存到三个变量中
            acc_i, SE_i, SP_i = calculate_each_image(SR, GT, threshold=0.5)
            
            # 累加结果到 acc、SE 和 SP 变量上
            acc += acc_i
            SE += SE_i
            SP += SP_i
            length += images.size(0)

        acc = acc / length
        SE = SE / length
        SP = SP / length
        val_loss = val_loss / length

        print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f' %(
            acc, SE, SP))

        # 保存模型checkpoint
        if epoch % 50 == 0:
            checkpoint_file = os.path.join(checkpoint_dir, '{}_{}_epoch{}.pth'.format(self.model_type,self.pre_type, epoch + 1))
            torch.save(self.unet.state_dict(), checkpoint_file)

        # 保存验证集图像
        if epoch % 10 == 0:
            torchvision.utils.save_image(images.data.cpu(),
                                         os.path.join(self.result_path,
                                                      '%s_valid_%d_image.png' % (self.model_type, epoch + 1)))
            torchvision.utils.save_image(SR.data.cpu(),
                                         os.path.join(self.result_path,
                                                      '%s_valid_%d_SR.png' % (self.model_type, epoch + 1)))
            torchvision.utils.save_image(GT.data.cpu(),
                                         os.path.join(self.result_path,
                                                      '%s_valid_%d_GT.png' % (self.model_type, epoch + 1)))
        return val_loss

    
    def train(self):
        train_losses = []
        val_losses = []

        unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' % (
            self.model_type, self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))
        if os.path.isfile(unet_path):
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
            lr = self.lr

        for epoch in range(self.num_epochs):
            gpu_id = 1
            torch.cuda.set_device(gpu_id)
            self.unet.to(self.device)
            self.unet.train(True)
            epoch_loss = 0
            acc = 0.
            SE = 0.
            SP = 0.
            length = 0
            if self.loss_type=="Dice":
                loss_function = L.DiceLoss(mode='binary')
            if self.loss_type=="BCE":
                loss_function = torch.nn.BCELoss()
            if self.loss_type == "mixed":
                bce_loss = torch.nn.BCELoss()
                dice_loss = L.DiceLoss(mode='binary')
                loss_function = lambda inputs, targets: bce_loss(inputs, targets) + dice_loss(inputs, targets)


            for i, (images, GT) in enumerate(self.train_loader):
                images = images.to(self.device)
                GT = GT.to(self.device)

                SR = self.unet(images)
                SR_probs = torch.sigmoid(SR)

                loss = loss_function(SR_probs, GT)
                epoch_loss += loss.item()

                self.reset_grad()
                loss.backward()
                self.optimizer.step()

                # 调用 calculate_each_image 函数，并将返回结果保存到三个变量中
                acc_i, SE_i, SP_i = calculate_each_image(SR, GT, threshold=0.5)
                
                # 累加结果到 acc、SE 和 SP 变量上
                acc += acc_i
                SE += SE_i
                SP += SP_i
                length += images.size(0)
                
            acc = acc / length
            SE = SE / length
            SP = SP / length
            
            train_losses.append({"loss":epoch_loss / length,"acc":acc,"SE":SE,"SP":SP})
            print(
                'Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP:%.4f'%(epoch + 1, self.num_epochs,epoch_loss,acc, SE, SP))

            if (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
                lr -= (self.lr / float(self.num_epochs_decay))
                self.update_lr(lr)
                print('Decay learning rate to lr: {}.'.format(lr))
            
            # 调用验证函数并获取验证损失
            val_loss = self.val(epoch)
            val_losses.append(val_loss)
        
            # Early stopping机制
            if acc+SE < self.best_val_loss:
                self.best_val_loss = acc+SE
                self.early_stop_count = 0
                # 保存最优模型
                best_model_path = os.path.join(self.model_path, 'best_model.pth')
                torch.save(self.unet.state_dict(), best_model_path)
                print('Best model saved with validation loss: {:.4f}'.format(val_loss))
            else:
                self.early_stop_count += 1
                print('Early stopping count: {}'.format(self.early_stop_count))
        
            if self.early_stop_count >= self.patience:
                print('Early stopping triggered')
                #break

    
        # 保存最后一个epoch的模型作为最终模型
        final_model_path = os.path.join(self.model_path, 'final_model.pth')
        torch.save(self.unet.state_dict(), final_model_path)
        # 保存训练损失到JSON文件
        with open(os.path.join(self.result_path, 'train_save.json'), 'w') as f:
            json.dump(train_losses, f)
        
        # 保存验证损失到JSON文件
        with open(os.path.join(self.result_path, 'val_save.json'), 'w') as f:
            json.dump(val_losses, f)
    
        # 释放未使用的内存
        del loss
        torch.cuda.empty_cache()
        
    def load_weights(self, weights_path):
        # 加载预训练权重
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))  # 加载权重
        self.unet.load_state_dict(checkpoint)  # 加载模型参数

    
    def test(self,best_model_path):
        """Evaluate the model on the test dataset."""
        self.unet.train(False)
        self.unet.eval()
        best_model_path = self.model_path+best_model_path
        print(best_model_path)
        if best_model_path:
            self.load_weights(best_model_path)
        else:
            return best_model_path+"not ex"
        for i, (images, GT) in enumerate(self.test_loader):
            images = images.to(self.device)
            GT = GT.to(self.device)
            SR = torch.sigmoid(self.unet(images))
    
            # 保存图像
            torchvision.utils.save_image(images.data.cpu(),
                                         os.path.join(self.result_path, '%s_test_%d_image.png' % (self.model_type, i + 1)))
            torchvision.utils.save_image(SR.data.cpu(),
                                         os.path.join(self.result_path, '%s_test_%d_SR.png' % (self.model_type, i + 1)))
            torchvision.utils.save_image(GT.data.cpu(),
                                         os.path.join(self.result_path, '%s_test_%d_GT.png' % (self.model_type, i + 1)))
            
            
            # 保存 GT 图像为二进制文件
            gt_binary_path = os.path.join(self.result_path, '%s_test_%d_GT.bin' % (self.model_type, i + 1))
            torch.save(GT, gt_binary_path)
    
            # 保存 SR 图像矩阵为二进制文件
            sr_binary_path = os.path.join(self.result_path, '%s_test_%d_SR.bin' % (self.model_type, i + 1))
            torch.save(SR, sr_binary_path)
    
            print("GT and SR images saved as binary files successfully.")
