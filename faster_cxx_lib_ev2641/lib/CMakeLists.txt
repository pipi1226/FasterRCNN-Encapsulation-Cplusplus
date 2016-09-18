cmake_minimum_required (VERSION 2.8)

SET (SRC_LIST faster_rcnn.cpp)
include_directories ( /usr/caffe/py-faster-rcnn-master/caffe-fast-rcnn/include
    /usr/caffe/py-faster-rcnn-master/lib/nms
    /usr/local/include
    /usr/include/python2.7
    /usr/include/atlas 
    )

add_library(faster_rcnn SHARED ${SRC_LIST})


