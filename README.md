# AppleSort
For apple detection run:

 ./checkModel.py --model_path appleDetectModel.h5 --classes apple_1class.txt  --images_dir AppleTest/Apple/
 
For apple classification run:

 ./checkModel.py --model_path appleClassiftcnModel.h5 --classes apple_classes.txt  --images_dir AppleTest/Apple/
 
 
 Hog + Pca + SVM('rbf') model (300X300) result:
 
 Prob treshhold, Precision, Recall, F1_score
      0.3           0.42     0.72     0.57 
      0.4           0.42     0.72     0.57
      0.5           0.44     0.72     0.54
      0.6           0.45     0.71     0.55
      0.7           0.47     0.71     0.58
      0.8           0.56     0.52     0.58
      0.9           0.56     0.57     0.58
 
 avg fps: 0.9s,
 AP = (0.62*5 + 0.56 + 0.41 + 0.5*4)/11 = 0.552
 
 Based yolo tiny v3 (416X416) network:
 
 Prob treshhold, Precision, Recall, F1_score
      0.3           0.79     0.65     0.72 
      0.4           0.82     0.52     0.57
      0.5           0.83     0.43     0.56
      0.6           0.91     0.36     0.51
      0.7           0.91     0.25     0.4
      0.8           0.85     0.14     0.25
      0.9           0.8      0.1      0.18

avg fps: 0.03s,
AP = (0.8*2 + 0.85 + 0.91 + 0.91 + 0.83 + 0.82 + 0.79*4) / 11 = 0.825

#bash commands

#learn hog + svm model:
./hogpcasvmlearn.py --train_data apple_data/ --result_path svmModel.joblib --meta_file_path apple_meta.txt --pca_result_path pcaModel.joblib

#detect hog svm model 
./hogpcasvmdetect.py --test_data AppleTest/Apple/ --model_path svmModel.joblib --pca_model_path pcaModel.joblib --detection_result_file hog+pca+svm_res.txt 

#detect yolo model
./yolodetect.py --images_dir AppleTest/Apple/ --model_path detectAppleYolo.h5 --class apple_1class.txt --result_file yolo_res.txt

#validate hog+svm model
./validate.py --ground_truth_file ground_truth.txt --test_result_file hog+pca+svm_res.txt

#validate yolo model
./validate.py --ground_truth_file ground_truth.txt --test_result_file yolo_res.txt
