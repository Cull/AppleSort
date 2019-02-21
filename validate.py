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
        '--ground_truth_file', required=True, help='ground truth file'
    )

    parser.add_argument(
        '--test_result_file', required=True, help='result file'
    )

    FLAGS = parser.parse_args()

    ground_truth_file = FLAGS.ground_truth_file
    result_file = FLAGS.test_result_file
    input_shape = (150,150)

    with open(ground_truth_file) as f:
        linesGT = f.readlines()

    with open(result_file) as f:
        linesRF = f.readlines()

    TP, FP, FN = validate(linesGT, linesRF)

    Recall = TP / (TP + FN)

    Precision = TP / (TP + FP)

    print("Recall --- " + str(Recall), "Precision --- " + str(Precision))

    print("F1score --- ", (2*Recall*Precision)/(Recall + Precision))

def union(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[2], b[2]) - x
    h = max(a[3], b[3]) - y
    return [x, y, x+w, y+h]

def intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[2], b[2]) - x
    h = min(a[3], b[3]) - y
    if w<0 or h<0: return [0,0,0,0]
    return [x, y, x+w, y+h]

def area(a):
    return (a[2]-a[0])*(a[3]-a[1])

def validate(linesGT, linesRF):

    TP, FP, FN = [0, 0, 0]

    print("validateing")

    if len(linesGT) != len(linesRF):
        print("ERROR: meta files have different sizes")
        exit(0)

    i = 0

    for i in range(len(linesRF)):
        lineT = linesGT[i]
        lineP = linesRF[i]
        cur_meta_t = lineT.split()
        cur_meta_p = lineP.split()
        file_t = cur_meta_t[0]
        file_p = cur_meta_p[0]

        if file_p != file_t:
            print(file_t, file_p)
            print("ERROR: invalid meta files format")
            exit(0)

        rects_gt = []
        rects_pt = []
        cur_meta_t = cur_meta_t[1:]
        cur_meta_p = cur_meta_p[1:]

        for meta_t in cur_meta_t:
            meta_t = meta_t.split(',')
            rect_t = [int(meta_t[0]), int(meta_t[1]), int(meta_t[2]), int(meta_t[3])]
            isFinded = False
            for meta_p in cur_meta_p:
                meta_p = meta_p.split(',')
                rect_p = [int(meta_p[0]), int(meta_p[1]), int(meta_p[2]), int(meta_p[3])]
                if area(intersection(rect_t, rect_p)) >  4 * area(union(rect_t,rect_p)) / 11:
                    isFinded = True
                    break
            if isFinded:
                TP += 1
            else:
                FN += 1

        for meta_p in cur_meta_p:
            meta_p = meta_p.split(',')
            rect_p = [int(meta_p[0]), int(meta_p[1]), int(meta_p[2]), int(meta_p[3])]
            isFinded = False
            for meta_t in cur_meta_t:
                meta_t = meta_t.split(',')
                rect_t = [int(meta_t[0]), int(meta_t[1]), int(meta_t[2]), int(meta_t[3])]
                if area(intersection(rect_t, rect_p)) > 4 * area(union(rect_t,rect_p)) / 11:
                    isFinded = True
                    break
            if not isFinded:
                FP += 1

    return TP, FP, FN

if __name__ == '__main__':
    _main()
