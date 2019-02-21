#!/usr/bin/python3

from os import listdir
from os.path import isfile, join

import argparse
import cv2
import numpy

from model import *

from PIL import Image

import time

if __name__ == '__main__':
# class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='path to model'
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str, required=True,
        help='path to class definitions, default '
    )

    parser.add_argument(
        '--images_dir', required=True, help='images path to detect'
    )

    parser.add_argument(
        '--result_file', required=True, help='detection result file'
    )

    FLAGS = parser.parse_args()
    mypath = FLAGS.images_dir
    model = YOLO(**vars(FLAGS))
    detection_result_file = FLAGS.result_file

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]

    dtime_avg = 0
    size = len(onlyfiles)

    #clean file
    f = open(detection_result_file, 'w')
    f.close()

    for n in range(0, len(onlyfiles)):
        image = cv2.imread(join(mypath, onlyfiles[n]))

        print('proceeded ' + str(n) + ' images, ' + str(len(onlyfiles) - n) + ' left to go.')

        cv2_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        start_time = time.time()
        pil_image, classes_names, detectedRects = model.detect_image(pil_im)
        dtime = time.time() - start_time
        dtime_avg += dtime/size
        print("detection time --- %s seconds ", dtime)
        f = open(detection_result_file, "a")
        f.write(onlyfiles[n])

        for rect in detectedRects:
            f.write(" " + str(rect[0]) + ',' + str(rect[1]) + ',' + str(rect[2]) +
                    ',' + str(rect[3]) + ',' + str(rect[4]))
            cv2.rectangle(image,(rect[0],rect[1]),(rect[2],rect[3]),(0,255,0),2)

        f.write('\n')
        f.close()

    print("average detection time --- %s seconds ", dtime_avg)



