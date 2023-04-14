import argparse as ap
import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
args = vars(parser.parse_args())

# Get the training classes names and store them in a list
train_path = args["trainingSet"]
training_names = os.listdir(train_path) # 不同类数据采用 label 作为文件夹名称
print(training_names)

# dataload 
image_paths = []
image_classes = []
class_id = 0
for training_name in training_names: # 遍历每一个类 数据文件夹
    dir = os.path.join(train_path, training_name)
    class_path = imutils.imlist(dir)
    image_paths += class_path
    image_classes += [class_id] * len(class_path) # 设置标签,用数字替换原始 label,可视化的时候再映射回去
    class_id += 1

des_list = []
orb = cv2.ORB_create()
for image_path in tqdm(image_paths, desc='Processing'):
    im = cv2.imread(image_path)
    kpts, des  = orb.detectAndCompute(im, None)
    des = des.astype('float32')
    des_list.append((image_path, des))   
    
# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))  

# Perform k-means clustering
k = 100
print("Start k-means: %d words, %d key points" %(k, descriptors.shape[0]))
voc, variance = kmeans(descriptors, k, 1) # # 质心, 平均距离

# Calculate the histogram of features
im_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
stdSlr = StandardScaler().fit(im_features)  # 针对每一个特征维度,计算 均值和方差
im_features = stdSlr.transform(im_features) # 根据均值和方差计算 标准化

# Train the Linear SVM
clf = LinearSVC()
clf.fit(im_features, np.array(image_classes))

# Save the SVM:  classifier, class name, 均值和方差, k-means, 
joblib.dump((clf, training_names, stdSlr, k, voc), "../vocabulary/voc_cls.pkl", compress=3)    
    
