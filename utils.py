import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

class ImagePreprocessor:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)

    def select_green_channel(self):
        return self.image[:, :, 1]

    def scale_image(self, image, target_size=(1000, 1000)):
        target_size = (1000, 1000)
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)

        return resized_image

    def map_to_full_range(self, image):
        min_val, max_val = np.percentile(image, (1, 99))
        mapped_image = np.clip(image, min_val, max_val)
        mapped_image = ((mapped_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        return mapped_image

    def saturate_pixels(self, image):
        low_thresh, high_thresh = np.percentile(image, (1, 99))
        image[image < low_thresh] = low_thresh
        image[image > high_thresh] = high_thresh
        return image

    def create_roi_mask(self):
        red_channel = self.image[:, :, 2]
        a = self.scale_image(red_channel)
        roi_mask = (a > 20).astype(np.uint8) * 255

        return roi_mask

    def smooth_roi_edge(self, roi_mask, image, kernel_size=5):
        # 提取ROI的边界

        # 对边界进行平滑处理
        smoothed_roi_edge = cv2.GaussianBlur(roi_mask, (5, 5), 0)
        # 将平滑后的边界与ROI区域进行合并
        smoothed_roi = cv2.bitwise_and(image, image, mask=smoothed_roi_edge)
        return smoothed_roi


def showpic(image, title):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')


def show_all(public_image_path):
    # Public Image
    public_preprocessor = ImagePreprocessor(public_image_path)
    public_green_channel = public_preprocessor.select_green_channel()
    scaled_public_image = public_preprocessor.scale_image(public_green_channel)
    mapped_public_image = public_preprocessor.map_to_full_range(scaled_public_image)
    saturated_public_image = public_preprocessor.saturate_pixels(mapped_public_image)
    roi_mask_public = public_preprocessor.create_roi_mask()
    scaled_mask_public = public_preprocessor.smooth_roi_edge(roi_mask_public, saturated_public_image)
    # Display the images
    # plt.figure(figsize=(15, 10))
    #
    # # Public Image
    # plt.subplot(2, 7, 1)
    # showpic(public_preprocessor.image, 'Original Public Image')
    #
    # plt.subplot(2, 7, 2)
    # showpic(public_green_channel, 'Selected Green Channel')
    #
    # plt.subplot(2, 7, 3)
    # showpic(scaled_public_image, 'Scaled Image')
    #
    # plt.subplot(2, 7, 4)
    # showpic(mapped_public_image, 'Mapped to Full Range')
    #
    # plt.subplot(2, 7, 5)
    # showpic(saturated_public_image, 'Saturated Image')
    #
    # plt.subplot(2, 7, 6)
    # showpic(roi_mask_public, 'ROI Mask')
    #
    # plt.subplot(2, 7, 7)
    # showpic(scaled_mask_public, 'Smoothed Image')
    return scaled_mask_public


# Example usage:
# public_image_path = '9.jpg'
# preproessed_public = show_all(public_image_path)
# private_image_path = 'private.JPG'
# preproessed_private = show_all(private_image_path)
# image_path = "/Users/hexue/pyproject/MA_seg/14. E-optha/e_optha_MA/MA/E0000043/DS000DGS.JPG"
# imageaa = show_all(image_path)

# private_green_channel = private_preprocessor.select_green_channel()
# scaled_private_image = private_preprocessor.scale_image()

# smoothed_public_image = public_preprocessor.smooth_roi_edge(scaled_public_image, roi_mask_public)

# # Private Image
# private_preprocessor = ImagePreprocessor(private_image_path)
# private_green_channel = private_preprocessor.select_green_channel()
# scaled_private_image = private_preprocessor.scale_image()
# mapped_private_image = private_preprocessor.map_to_full_range(private_green_channel)
# saturated_private_image = private_preprocessor.saturate_pixels(mapped_private_image)
# roi_mask_private = private_preprocessor.create_roi_mask()
# smoothed_private_image = private_preprocessor.smooth_roi_edge(scaled_private_image, roi_mask_private)

import cv2
import numpy as np
import os


def scale_space_representation(image, num_scales):
    # 初始化尺度空间列表
    scale_space_images = []

    # 将图像作为初始尺度空间图像的一部分
    scale_space_images.append(image)

    # 生成其他尺度空间图像
    for i in range(1, num_scales):
        # 定义尺度参数𝑡
        t = i * np.sqrt(2) / 4
        # 使用方差为sqrt(2)/2的高斯核
        sigma = np.sqrt(2 * t)
        kernel_size = 3
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        # 进行高斯滤波
        smoothed_image = cv2.filter2D(scale_space_images[i - 1], -1, kernel)
        # 添加到尺度空间列表
        scale_space_images.append(smoothed_image)
    return scale_space_images
def process_scaled_images(scaled_images, num_scales_to_keep):
    # 仅保留较高尺度的图像
    scaled_images_to_process = scaled_images[num_scales_to_keep:]
    return scaled_images_to_process


import numpy as np
import cv2


# 计算Hessian矩阵的特征值
def calculate_eigenvalues(hessian_matrix):
    eigenvalues, _ = np.linalg.eig(hessian_matrix)
    return eigenvalues


# 在给定的点(x, y)处计算Hessian矩阵
def calculate_hessian_matrix(images, x, y, t):
    image = images[t]
    # 计算图像的二阶偏导数
    Lxx = cv2.Sobel(image, cv2.CV_64F, 2, 0, ksize=5)
    Lyy = cv2.Sobel(image, cv2.CV_64F, 0, 2, ksize=5)
    Lxy = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)

    # 计算Hessian矩阵的元素
    H = np.array([[Lxx[x, y], Lxy[x, y]], [Lxy[x, y], Lyy[x, y]]])
    return H


def filter_top_n_brightest_pixels(image, n):
    # 将图像展平并获取灰度值
    flattened_image = image.flatten()
    # 找到最亮的n个像素的灰度值
    brightest_values = np.sort(flattened_image)[-n:]
    # print(brightest_values)
    # 创建新的图像矩阵
    filtered_image = np.zeros_like(flattened_image)

    # 对每个像素点进行判断，保留大于阈值的像素
    for i, pixel_value in enumerate(flattened_image):
        if pixel_value in brightest_values:
            filtered_image[i] = pixel_value

    # 将图像重塑回原始形状
    filtered_image = filtered_image.reshape(image.shape)

    return filtered_image



# 在给定的图像上计算Hessian矩阵的最大特征值和最小特征值矩阵
# 在给定的图像上计算Hessian矩阵的最大特征值和最小特征值矩阵
def calculate_eigenvalue_matrices(images):
    max_eigenvalue_matrix = np.zeros(images[0].shape[:2])
    min_eigenvalue_matrix = np.zeros(images[0].shape[:2])
    final_max = []
    final_min = []
    for t, image in enumerate(images):
        Lxx = cv2.Sobel(image, cv2.CV_64F, 2, 0, ksize=5)
        Lyy = cv2.Sobel(image, cv2.CV_64F, 0, 2, ksize=5)
        Lxy = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                H = np.array([[Lxx[i, j], Lxy[i, j]], [Lxy[i, j], Lyy[i, j]]])
                eigenvalues = calculate_eigenvalues(H)
                max_eigenvalue_matrix[i, j] = np.max(eigenvalues)
                min_eigenvalue_matrix[i, j] = np.min(eigenvalues)
        final_max.append(max_eigenvalue_matrix)
        final_min.append(min_eigenvalue_matrix)

    return final_max, final_min



import cv2
import matplotlib.pyplot as plt
import os


def generate_derivative_masks(num_masks):
    derivative_masks = []

    for i in range(1 + 1, num_masks + 2):
        # 构建零向量
        D = np.zeros(2 * i - 1)
        # 设置第一个和最后一个元素为 -1 和 1
        D[0] = -1
        D[-1] = 1
        # 将导数模板添加到列表中
        derivative_masks.append(D)

    return derivative_masks


def apply_derivative_mask(image, derivative_mask):
    # 使用 OpenCV 的 filter2D 函数应用导数模板
    filtered_image = cv2.filter2D(image, -1, derivative_mask)
    return filtered_image


# BW BW就是一个roi
# https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=2bedc8b6b8dd9f088f4a1a6e0fe13933018ce7d7
# mask 为了删除边缘，
# testimage = filtered_images[0][0]
import cv2

import matplotlib.pyplot as plt


def calculate_edge_magnitude(image):
    """
    计算图像的边缘幅度
    """
    # 使用高斯导数方法计算图像的边缘幅度
    # 计算 x 方向和 y 方向上的边缘
    x_edge = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    y_edge = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # 计算边缘幅度
    edge_magnitude = np.sqrt(x_edge ** 2 + y_edge ** 2)
    return edge_magnitude


def shrink_mask(mask, shrink_factor):
    """
    将掩码图像缩小
    """
    # 计算缩小后的图像大小
    new_height = int(mask.shape[0] * shrink_factor)
    new_width = int(mask.shape[1] * shrink_factor)

    # 使用双线性插值缩小图像
    shrunk_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return shrunk_mask


def select_strongest_edge_pixels(edge_magnitude, edge_percentage):
    """
    选择最强的边缘像素
    """
    # 计算边缘幅度图像的直方图
    hist, bins = np.histogram(edge_magnitude.flatten(), bins=256, range=[0, 256])
    cum_hist = np.cumsum(hist)

    # 计算要选择的像素数量
    total_pixels = edge_magnitude.shape[0] * edge_magnitude.shape[1]
    edge_pixels_threshold = total_pixels * edge_percentage / 100

    # 根据阈值选择最强的边缘像素
    selected_pixels_indices = np.where(cum_hist >= edge_pixels_threshold)[0]
    strongest_edge_pixels = [bins[idx] for idx in selected_pixels_indices]
    return strongest_edge_pixels


def create_roi_mask(image):
    roi_mask = (image > 25).astype(np.uint8) * 255
    return roi_mask


# 计算T矩阵
# 计算T矩阵
def calculate_T_matrix(image, min_eigenvalue_matrix):
    # 计算拉普拉斯变换的立方
    laplacian_cubed = np.power(cv2.Laplacian(image, cv2.CV_64F), 3)
    T_matrix = laplacian_cubed - np.minimum(min_eigenvalue_matrix, laplacian_cubed)
    return T_matrix


# 计算 A 矩阵
def calculate_A_matrix(T_matrix, max_eigenvalue_matrix, min_eigenvalue_matrix, laplacian_matrix):
    # 计算分子部分
    numerator = (np.square(max_eigenvalue_matrix) / np.square(min_eigenvalue_matrix)) * T_matrix
    # 计算分母部分
    denominator = laplacian_matrix
    # 计算 A 矩阵
    A_matrix = numerator / denominator

    return A_matrix


def calculate(filtered_images,all_max_eigenvalue_matrix, all_min_eigenvalue_matrix):
    final_accumulated_image=[]
    # 遍历每张图片
    eachscale_curvature=[]
    accumulated_image=None
    # 调用函数计算特征值矩阵
    # all_max_eigenvalue_matrix = []
    # all_min_eigenvalue_matrix = []
    for eachscale in range(0,5):
        curvatureeach_d=[]
        for each_d in range(0,3):
            j=each_d
            i=eachscale
            # print(i,j)
            image = filtered_images[i][j]
            # print(len(all_max_eigenvalue_matrix[i][j]))
            max_eigenvalue_matrix=all_max_eigenvalue_matrix[i][j]
            min_eigenvalue_matrix=all_min_eigenvalue_matrix[i][j]
            T_matrix = calculate_T_matrix(image, min_eigenvalue_matrix)
            curvature=(max_eigenvalue_matrix+min_eigenvalue_matrix)/2
            curvatureeach_d.append(curvature)
            laplacian_matrix=np.power(cv2.Laplacian(image, cv2.CV_64F), 3)
            A_matrix = calculate_A_matrix(T_matrix, max_eigenvalue_matrix, min_eigenvalue_matrix, laplacian_matrix)
            # 如果accumulated_image为空，则将当前A_matrix赋值给它，否则将A_matrix叠加到accumulated_image上
            if accumulated_image is None:
                accumulated_image = A_matrix
            else:
                accumulated_image *= A_matrix
        eachscale_curvature.append(curvatureeach_d)
        final_accumulated_image.append(accumulated_image)
        #返回计算的每个图片的值和最后叠加图像的值
    return eachscale_curvature,final_accumulated_image



def preprocess_image(image):
    # 将图像归一化到 [0, 1] 区间
    normalized_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    # 直方图均衡化
    equalized_image = cv2.equalizeHist((normalized_image * 255).astype(np.uint8))

    return equalized_image


def multi_scale_vessel_detector(images):
    # 对每个尺度的图像进行预处理
    preprocessed_images = [preprocess_image(image) for image in images]

    # 计算 multi-scale vessel detector 𝑉1(𝐱)
    vessel_detector = np.zeros_like(preprocessed_images[0], dtype=np.float64)
    for image in preprocessed_images[2:]:
        vessel_detector += image

    return vessel_detector


# 选择给定百分比的输入序列中的顶部值
def top_percentage(image, percentage):
    # 将图像展平并排序
    flattened_image = np.ravel(image)
    sorted_values = np.sort(flattened_image)
    # 计算要保留的值的数量
    num_values_to_keep = int(len(sorted_values) * (percentage / 100))
    # 选择顶部百分比的值
    top_values = sorted_values[-num_values_to_keep:]
    # 将其重新形状回原始图像大小
    return top_values.reshape(image.shape)

# 将一组增强的图像叠加起来
def combine_images(images):
    return np.sum(images, axis=0)