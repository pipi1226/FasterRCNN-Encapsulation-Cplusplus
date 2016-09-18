This project is modified by https://github.com/YihangLou/FasterRCNN-Encapsulation-Cplusplus

Thanks to his SourceCode.

This version is modified and it can be used in cpu mode without CUDA.
It couldn't compile because function [ _nms() ] is in gpu_nms.so, my computer didn't have this so.

modified places:

-------------------------------------
Head File

lib/faster_rcnn.hpp
Line 4-6:
Add Macro --> #define CPU_ONLY 1
-------------------------------------



-------------------------------------
CPP File

-------------------------------------
lib/faster_rcnn.cpp:


Line 37:
Add structure --> typedef struct _ST_SCORE ...
----------------------------

Line 155:
Add function --> 
cpu_nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh)


----------------------------

Line 421:
Modified function --> Detect(cv::Mat & cv_img, vector<cv::Rect> & detection_result )

-------------------------------------


