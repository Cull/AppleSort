#!/usr/bin/python3
import argparse
import cv2

from os import listdir
from os.path import isfile, join

from skimage.feature import hog

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.externals import joblib

import time

def _main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--test_data', required=True, help='Path to train data'
    )

    parser.add_argument(
        '--model_path', required=True, help='Path to model'
    )

    parser.add_argument(
        '--pca_model_path', required=True, help='Path to pca model'
    )

    parser.add_argument(
        '--detection_result_file', required=True, help='Result file'
    )

    FLAGS = parser.parse_args()

    data_path = FLAGS.test_data
    model_path = FLAGS.model_path
    pca_model_path = FLAGS.pca_model_path
    detection_result_file = FLAGS.detection_result_file

    model_shape = (150,150)
    loaded_model = joblib.load(model_path)
    pca_model = joblib.load(pca_model_path)

    #clean file
    f = open(detection_result_file, 'w')
    f.close()

    detect_apple(model_shape, data_path, loaded_model, pca_model, detection_result_file)


def detect_apple(model_shape, data_path, model, pca_model, detection_result_file):

    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
    dtime_avg = 0
    size = len(onlyfiles)
    for n in range(0, size):
        image = cv2.imread(join(data_path, onlyfiles[n]))

        print('proceeded ' + str(n) + ' images, ' + str(len(onlyfiles) - n) + ' left to go.')

        detectedRects, dtime = slidingWindow(model_shape, image, model, pca_model)
        dtime_avg += dtime * (1/size)
        f = open(detection_result_file, "a")
        f.write(onlyfiles[n])

        for rect in detectedRects:
            f.write(" " + str(rect[0]) + ',' + str(rect[1]) + ',' + str(rect[2]) +
                    ',' + str(rect[3]) + ',' + str(rect[4]))
            #cv2.rectangle(image,(rect[0],rect[1]),(rect[2],rect[3]),(0,255,0),2)

        f.write('\n')
        f.close()
    print("average detection time --- %s seconds ", dtime_avg)

    return detectedRects

def slidingWindow(model_shape, image, model, pca_model):
    start_time = time.time()
    height, width, channels = image.shape
    scales = [0.25, 0.35, 0.5, 0.6, 0.8]

    detectedRects = []

    maxSize = width if width > height else height

    for scale in scales:

        windowWidth = int(maxSize * scale)
        windowHeight = int(maxSize * scale)

        if windowHeight < 40 or windowWidth < 40:
            continue

        x = 0
        y = 0

        while(x + windowWidth < width):
            y = 0
            while(y + windowHeight < height):
                x2 = x+windowWidth if x+windowWidth < width else width - 1
                y2 = y+windowHeight if y+windowHeight < height else height - 1
                crop_image = image[y:y2,x:x2]
                #cv2.imshow("cropedImage", crop_image)
                resizedImage = cv2.resize(crop_image, model_shape)
                greyIm = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY);
                fd = hog(greyIm, orientations=8, pixels_per_cell=(16, 16),
                         cells_per_block=(1, 1), block_norm='L2-Hys', feature_vector=True)
                #cv2.imshow("hog", hog_image)
                #cv2.imshow("resizedImage", greyIm)
                #cv2.waitKey(20)
                imageVec = pca_model.transform(fd.reshape(1,-1))
                probability = model.predict_proba(imageVec)
                c = model.predict(imageVec)
                if c == 0 and probability[0,0] > 0.9:
                    detectedRects.append([x,y,int(x+windowWidth),int(y+windowHeight),0])
                    y += int(windowHeight)
                else:
                    y += int(windowHeight / (20 * scale))
            x += int(windowWidth / (20 * scale))
    dtime = time.time() - start_time
    print("detection time --- %s seconds ---" % dtime)
    return filterRects(detectedRects), dtime 

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

def filterRects(detectedRects):
    rects = []
    for new_rect in detectedRects:
        addToRects = True
        for rect in rects:
            minrect = new_rect if area(new_rect) < area(rect) else rect
            if area(intersection(new_rect, rect)) > 3/5 * area(union(new_rect,rect)):
                if minrect == rect:
                    rects.remove(rect)
                else:
                    addToRects = False
                    break
        if addToRects == True:
            rects.append(new_rect)
    #remove incapsulated rects
    resrects = []
    for check_rect in rects:
        isIncapsulated = False
        for rect in rects:
            if check_rect != rect and area(intersection(check_rect, rect)) == area(rect):
                isIncapsulated = True
        if not isIncapsulated:
            resrects.append(check_rect)
    return resrects

if __name__ == '__main__':
    _main()