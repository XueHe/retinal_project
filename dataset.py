import os
import argparse
import random
import shutil
from shutil import copyfile
from misc import printProgressBar


def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s'%dir_path)
    os.makedirs(dir_path)
    print('Create path - %s'%dir_path)

def main(config,input_filenames):
    for filei in input_filenames:
        origin_data_path=config.path+filei+"/"
        train_path="./dataset/"+filei+"_train/"
        train_GT_path="./dataset/"+filei+"_train_GT/"
        valid_path="./dataset/"+filei+"_valid/"
        valid_GT_path="./dataset/"+filei+"_valid_GT/"
        test_path="./dataset/"+filei+"_test/"
        test_GT_path="./dataset/"+filei+"_test_GT/"
        
        rm_mkdir(train_path)
        rm_mkdir(train_GT_path)
        rm_mkdir(valid_path)
        rm_mkdir(valid_GT_path)
        rm_mkdir(test_path)
        rm_mkdir(test_GT_path)
    
        filenames = os.listdir(origin_data_path)
        data_list = []
        GT_list = []
        for filename in filenames:
            ext = os.path.splitext(filename)[-1]
            if ext =='.jpg':
                filename = filename.split('_')[-1][:-len('.jpg')]
                data_list.append(''+filename+'.jpg')
                GT_list.append(''+filename+'.png')
            if ext =='.JPG':
                filename = filename.split('_')[-1][:-len('.JPG')]
                data_list.append(''+filename+'.JPG')
                GT_list.append(''+filename+'.png')
        num_total = len(data_list)
        num_train = int((config.train_ratio/(config.train_ratio+config.valid_ratio+config.test_ratio))*num_total)
        num_valid = int((config.valid_ratio/(config.train_ratio+config.valid_ratio+config.test_ratio))*num_total)
        num_test = num_total - num_train - num_valid
    
        print('\nNum of train set : ',num_train)
        print('\nNum of valid set : ',num_valid)
        print('\nNum of test set : ',num_test)
    
        Arange = list(range(num_total))
        random.shuffle(Arange)
    
        for i in range(num_train):
            idx = Arange.pop()
            
            src = os.path.join(origin_data_path, data_list[idx])
            dst = os.path.join(train_path,data_list[idx])
            copyfile(src, dst)
            
            src = os.path.join(config.origin_GT_path, GT_list[idx])
            dst = os.path.join(train_GT_path, GT_list[idx])
            copyfile(src, dst)
    
            printProgressBar(i + 1, num_train, prefix = 'Producing train set:', suffix = 'Complete', length = 50)
            
    
        for i in range(num_valid):
            idx = Arange.pop()
    
            src = os.path.join(origin_data_path, data_list[idx])
            dst = os.path.join(valid_path,data_list[idx])
            copyfile(src, dst)
            
            src = os.path.join(config.origin_GT_path, GT_list[idx])
            dst = os.path.join(valid_GT_path, GT_list[idx])
            copyfile(src, dst)
    
            printProgressBar(i + 1, num_valid, prefix = 'Producing valid set:', suffix = 'Complete', length = 50)
    
        for i in range(num_test):
            idx = Arange.pop()
    
            src = os.path.join(origin_data_path, data_list[idx])
            dst = os.path.join(test_path,data_list[idx])
            copyfile(src, dst)
            
            src = os.path.join(config.origin_GT_path, GT_list[idx])
            dst = os.path.join(test_GT_path, GT_list[idx])
            copyfile(src, dst)
    
    
            printProgressBar(i + 1, num_test, prefix = 'Producing test set:', suffix = 'Complete', length = 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model hyper-parameters
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--valid_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--path', type=str,default="./dataset/eopt/")
    parser.add_argument('--origin_GT_path', type=str,default="./dataset/eopt/preprocessed_mask")
    
    

    config = parser.parse_args()
    input_filenames = [
    'preimage',
    'pre1image',
    'pre2image',
    'pre3image',
    'pre4image',
    'pre5image',
    'pre6image',
    'pre7image',
    'pre14image',
    'pre16image',  # Modified for preprocessing 16 output path
    'pre25image',  # Modified for preprocessing 25 output path
    'pre27image'   # Modified for preprocessing 27 output path
]

    print(config)
    main(config,input_filenames)