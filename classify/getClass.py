import argparse as ap
import time
import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn import preprocessing

def main():
    clf, classes_names, stdSlr, k, voc = joblib.load("../vocabulary/voc_cls.pkl")
    # /media/ubuntu/zoro/ubuntu/data/train/station/test/001801.png
    parser = ap.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--testingSet", help="Path to testing Set")
    group.add_argument("-i", "--image", help="Path to image")
    parser.add_argument('-v',"--visualize", action='store_true')
    args = vars(parser.parse_args())

    start_time = time.time()

    # Get the path of the testing image(s) and store them in a list
    image_paths = []
    if args["testingSet"]:
        test_path = args["testingSet"]
        try:
            testing_names = os.listdir(test_path)
        except OSError:
            print("No such directory {}\nCheck if the file exists".format(test_path))
            exit()
        for testing_name in testing_names:
            dir = os.path.join(test_path, testing_name)
            class_path = imutils.imlist(dir)
            image_paths += class_path
    else:
        image_paths = [args["image"]]
        
    orb = cv2.ORB_create()
    des_list = []
    for image_path in image_paths:
        im = cv2.imread(image_path)
        kpts, des  = orb.detectAndCompute(im, None)
        des = des.astype('float32')
        des_list.append((image_path, des))   
        
    # descriptors = des_list[0][1]
    # for image_path, descriptor in des_list[0:]:
    #     descriptors = np.vstack((descriptors, descriptor)) 

    test_features = np.zeros((len(image_paths), k), "float32")
    for i in range(len(image_paths)):
        words, distance = vq(des_list[i][1],voc)
        for w in words:
            test_features[i][w] += 1

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')
    test_features = test_features * idf
    test_features = preprocessing.normalize(test_features, norm='l2')
    # Scale the features
    test_features = stdSlr.transform(test_features)

    # Perform the predictions 
    predictions =  [classes_names[i] for i in clf.predict(test_features)]    
    scores =  [i for i in clf.decision_function(test_features)] # 决策函数, 返回

    # scores =  [np.max(i) for i in clf.decision_function(test_features)]
    print(predictions[0], scores)

    end_time = time.time()
    print("run time: {:.2f}s, FPS:{:.1f}".format(end_time-start_time, 1.0 / (end_time-start_time)))

    if args["visualize"]:
        for image_path, prediction, score in zip(image_paths, predictions, scores):
            image = cv2.imread(image_path)
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            pt = ( image.shape[1] // 10, image.shape[0] // 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, prediction + ":"+ str(round(score,3)), pt ,font, 1, [0, 255, 0], 2)
            cv2.imshow("Image", image)
            cv2.imwrite('result.png', image)
            # cv2.waitKey(3000)


main()