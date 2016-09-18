
#This part is used for compile faster_rcnn_demo.cpp
cmake_minimum_required (VERSION 2.8)

project (main_demo)

add_executable(main main.cpp)

include_directories ( /usr/caffe/py-faster-rcnn-master/caffe-fast-rcnn/include
    /usr/caffe/py-faster-rcnn-master/lib/nms
    /usr/local/include
    /usr/include/python2.7
    /usr/include/atlas 
    )

target_link_libraries(main /home/louxiu/caffe_Pro/caffe-gcc/faster_cxx_lib_ev2641/lib/libfaster_rcnn.so
    /usr/caffe/py-faster-rcnn-master/caffe-fast-rcnn/build/lib/libcaffe.so
    /usr/local/lib/libopencv_highgui.so 
    /usr/local/lib/libopencv_core.so 
    /usr/local/lib/libopencv_imgproc.so 
    /usr/local/lib/libopencv_imgcodecs.so
    /usr/lib/x86_64-linux-gnu/libglog.so
    /usr/lib/x86_64-linux-gnu/libboost_system.so
    /usr/lib/x86_64-linux-gnu/libboost_python.so
    /usr/lib/x86_64-linux-gnu/libpython2.7.so
    )


