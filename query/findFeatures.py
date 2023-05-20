import argparse as ap
import cv2
import numpy as np
import os
import pickle
from scipy.cluster.vq import vq, kmeans
from sklearn import preprocessing

# 1、首先提取对n幅图像分别提取SIFT特征.
# 2、然后对提取的整个SIFT特征进行 KMeans聚类 得到k个聚类中心作为视觉单词表（或者说是词典）.
# 3、最后对每幅图像以单词表为规范 对该幅图像的每一个SIFT特征点 计算它与单词表中每个单词的距离，最近的+1，便可得到该幅图像的码本。
#   每一幅图像就变成了一个与视觉词序列相对应的词频矢量。

# python searchFeatures.py -t /media/ubuntu/zoro/ubuntu/data/train/00/image_0
# /media/ubuntu/zoro/ubuntu/data/train/00/image_0
parser = ap.ArgumentParser() # 建立解析对象
parser.add_argument("-t", "--trainingSet", default= "/media/ubuntu/zoro/ubuntu/data/train/00/image_0", help="Path to Training Set", required="True")
args = vars(parser.parse_args()) # 实例
train_path = args["trainingSet"]
numWords = 1000

image_paths = []
training_names = os.listdir(train_path)
training_names.sort()
training_names = training_names[::5] # 原始数据太多,15 FPS 没必要区很多重复的东西
for training_name in training_names:
    image_path = os.path.join(train_path, training_name)
    image_paths += [image_path]

# 1, 提取存储SIFT特征
# stored descriptors
des_list = [] 
# fea_det = cv2.SIFT_create() 
orb = cv2.ORB_create()

for i, image_path in enumerate(image_paths):
    im = cv2.imread(image_path)
    print("Extract SIFT of %s image, %d of %d images" %(training_names[i], i, len(image_paths)))
    # kpts = fea_det.detect(im)
    # kpts, des = fea_det.compute(im, kpts)
    kpts, des  = orb.detectAndCompute(im, None)
    des = des.astype('float32')
    des_list.append((image_path, des))

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

# 2,聚类学习 “视觉词典（visual vocabulary）
print("Start k-means: %d words, %d key points" %(numWords, descriptors.shape[0]))
voc, variance = kmeans(descriptors, numWords, 1) # 质心, 平均距离

# Calculate the histogram of features
im_features = np.zeros((len(image_paths), numWords), "float32")
for i in range(len(image_paths)):
    # 将 voc(由kmeans生成)中的代码 指定给观测值 
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')


# Perform L2 normalization
im_features = im_features * idf
im_features = preprocessing.normalize(im_features, norm='l2')

with open('../vocabulary/voc_orb.pkl','wb') as f:
    pickle.dump((im_features, image_paths, idf, numWords, voc), f)

