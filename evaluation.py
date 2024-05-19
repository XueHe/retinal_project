import torch

# SR : Segmentation Result
# GT : Ground Truth

def calculate_each_image(SR,GT, threshold=0.5):
    gt_matrix = GT
    sr_matrix = SR > threshold
    
    # 将 SR 图像矩阵移到与 GT 矩阵相同的设备上
    sr_matrix = sr_matrix.to(gt_matrix.device)
    
    # 将 SR 图像矩阵转换为二值化矩阵（根据阈值0.5）
    sr_binary = torch.where(sr_matrix >= 0.5, torch.tensor(1,device=sr_matrix.device), torch.tensor(0, device=sr_matrix.device))
    
    # 计算真阳性、假阳性、真阴性、假阴性
    tp = torch.sum((gt_matrix == 1) & (sr_binary == 1)).item()
    fp = torch.sum((gt_matrix == 0) & (sr_binary == 1)).item()
    tn = torch.sum((gt_matrix == 0) & (sr_binary == 0)).item()
    fn = torch.sum((gt_matrix == 1) & (sr_binary == 0)).item()
        
    # 计算性能指标
    try:
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
    except:
        accuracy = (tp + tn) / ((tp + fp + tn + fn)+1)
        sensitivity = tp / ((tp + fn)+1)
        specificity = tn / ((tn + fp)+1)
    return accuracy, sensitivity, specificity
