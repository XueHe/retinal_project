relu_based_pre.ipynb 数据预处理
【处理epot数据，
eopt	pre1image	1次gs filter
eopt	pre2image	2次gs filter
eopt	pre3image	3次gs filter
eopt	pre4image	4次gs filter
eopt	pre5image	5次gs filter
eopt	pre6image	6次gs filter
eopt	pre7image	7次gs filter
eopt	pre14image	叠加1-4
eopt	pre16image	叠加1-6
eopt	pre25image	叠加2-5
eopt	pre27image	叠加2-7
eopt	preprocessed_mask	resize+binary】
python3 dataset.py 分割训练验证测试数据集
chmod +x run1.sh #train
./run1.sh
chmod +x run1.sh #test
./test1.sh
model_evaluation.ipynb 对模型测试集的表现做统计
image_show.ipynb 对test图片和vaild图片做层次叠加
