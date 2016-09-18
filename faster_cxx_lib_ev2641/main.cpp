#include "lib/faster_rcnn.hpp"
int main()
{
	string model_file = "/usr/caffe/py-faster-rcnn-master/models/pascal_voc/VGG16/faster_rcnn_alt_opt/faster_rcnn_test.pt";
	string weights_file = "/usr/caffe/py-faster-rcnn-master/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel";
	// /home/sky/ProgramFiles/py-faster-rcnn-master/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel
    	int GPUID=0;
    	int max_ret_num=30;

	Detector * handle = NULL;
    	EV2641_InitCarDetector(model_file.c_str(), weights_file.c_str(), GPUID , handle);
    	vector<cv::Rect> detection_result;
    	cv::Mat inputimage = cv::imread("img/horse1.jpg");
    	if(inputimage.empty())
    	{
        	std::cout<<"Can not get the image"<<endl;
        	return 0;
    	}
    	handle->Detect(inputimage, detection_result);

    	cout << "point vector size = " << detection_result.size() << endl;
    	for(int i=0;i < detection_result.size(); i++)
		{
        	cv::rectangle(inputimage,cv::Point(detection_result[i].x,detection_result[i].y),
                               cv::Point(detection_result[i].x + detection_result[i].width,detection_result[i].y + detection_result[i].height),
                               cv::Scalar(0,255,0));

    	}
    	cv::namedWindow("test-horse");
    	cv::imshow("test-horse", inputimage);
    	cv::imwrite("test.jpg",inputimage);
    	return 0;
}
