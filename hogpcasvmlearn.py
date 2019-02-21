#!/usr/bin/python3
import argparse
import cv2

from os import listdir
from os.path import isfile, join

from sklearn import svm

from skimage.feature import hog

from sklearn.decomposition import PCA
from sklearn.externals import joblib

import pickle

def _main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_data', required=True, help='Path to train data'
    )

    parser.add_argument(
        '--result_path', required=True, help='Path to save model'
    )

    parser.add_argument(
        '--meta_file_path', required=True, help='path to meta file'
    )

    parser.add_argument(
        '--pca_result_path', required=True, help='path to save pca model'
    )    

    FLAGS = parser.parse_args()

    data_path = FLAGS.train_data
    result_path = FLAGS.result_path
    meta_file_path = FLAGS.meta_file_path
    pca_result_path = FLAGS.pca_result_path

    input_shape = (300,300) 

    with open(meta_file_path) as f:
        files_meta_lines = f.readlines()

    num_examples = len(files_meta_lines)

    trainX, trainY = prepare_train_data(files_meta_lines, num_examples, input_shape, data_path)

    pca = PCA(n_components=100)

    pca.fit(trainX)
    trainX = pca.transform(trainX)

    clf = svm.SVC(kernel='rbf', gamma='scale', probability=True)
    clf.fit(trainX, trainY)

    print("model trained")

    joblib.dump(pca, pca_result_path)
    joblib.dump(clf, result_path)

    print(clf.score(trainX, trainY))

def prepare_train_data(files_meta_lines, num_examples, input_shape, data_path):
    trainX = []
    trainY = []
    for line in files_meta_lines:
        cur_meta = line.split()
        filepath = join(data_path, cur_meta[0])
        cur_rects = cur_meta[1:]
        if isfile(filepath):
            for rect in cur_rects:
                meta = rect.split(",")
                image = cv2.imread(filepath)
                height, width, channels = image.shape
                p1x = int(meta[0]) if int(meta[0]) > 0 else 0 
                p1y = int(meta[1]) if int(meta[1]) > 0 else 0
                p2x = int(meta[2]) if int(meta[2]) < width else width - 1
                p2y = int(meta[3]) if int(meta[3]) < height else height - 1
                cl = int(meta[4])
                print(p1x,p1y,p2x,p2y)
                crop_image = image[p1y:p2y, p1x:p2x]
                #cv2.imshow("cur", crop_image)
                resizedImage = cv2.resize(crop_image, input_shape)
                greyIm = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY);
                fd = hog(greyIm, orientations=8, pixels_per_cell=(16, 16),
                         cells_per_block=(1, 1), block_norm='L2-Hys', feature_vector=True)
                print(filepath, cl)
                #cv2.imshow("resizedImage", greyIm)
                #cv2.waitKey(100)
                trainX.append(fd)
                trainY.append(cl)
    print("data pretrained")
    return trainX, trainY

if __name__ == '__main__':
    _main()
