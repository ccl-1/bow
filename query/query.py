# -*- coding: utf-8 -*-
import argparse as ap
import cv2
import pickle
from scipy.cluster.vq import vq
from sklearn import preprocessing
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time


def main():
	# python query.py -i /media/ubuntu/zoro/ubuntu/data/train/00/image_0/000045.png
	parser = ap.ArgumentParser()
	parser.add_argument("-i", "--image", help="Path to query image", required="True")
	args = vars(parser.parse_args())

	start_time = time.time()
	image_path = args["image"]
	im_features, image_paths, idf, numWords, voc = pickle.load(open('../vocabulary/voc_orb.pkl','rb'))
	# fea_det = cv2.SIFT_create() 
	orb = cv2.ORB_create()

	des_list = []
	im = cv2.imread(image_path)
	# kpts = fea_det.detect(im)
	# kpts, des = fea_det.compute(im, kpts)
	kpts, des  = orb.detectAndCompute(im, None)
	des = des.astype('float32')

	des_list.append((image_path, des))

	# Stack all the descriptors vertically in a numpy array
	descriptors = des_list[0][1]
	test_features = np.zeros((1, numWords), "float32")
	words, distance = vq(descriptors,voc)
	for w in words:
		test_features[0][w] += 1

	# Perform Tf-Idf vectorization and L2 normalization
	test_features = test_features*idf
	test_features = preprocessing.normalize(test_features, norm='l2')
	
	score = np.dot(test_features, im_features.T)
	rank_ID = np.argsort(-score) #降序,返回索引

	end_time = time.time()
	print("run time: {:.2f}s, FPS:{:.1f}".format(end_time-start_time, 1.0 / (end_time-start_time)))


	plt.figure(figsize=(15,8))
	fig = plt.subplot(3,3,1)
	plt.imshow(im[:,:,::-1])
	plt.title("Query: "+image_path[-10:-4]) 
	plt.axis('off')
	# Visualize Query results of top 6 . 
	for i, ID in enumerate(rank_ID[0][0:6]):
		img = Image.open(image_paths[ID])
		plt.subplot(3,3,i+4)
		plt.imshow(img)
		plt.title(image_paths[ID][-10:-4] + ' :' 
		+ str(round(score[0][ID], 3)))
		plt.axis('off')
	plt.savefig('result.png')


if __name__ == "__main__":
      main()