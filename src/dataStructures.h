#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <opencv2/core.hpp>


struct DataFrame { // represents the available sensor information at the same time instance

    cv::Mat cameraImg; // camera image

    std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
    cv::Mat descriptors; // keypoint descriptors
    std::vector<cv::DMatch> kptMatches; // keypoint matches between previous and current frame
};

enum DetectorType
{
    SHITOMASI_Det,
    HARRIS_Det,
    FAST_Det,
    BRISK_Det,
    ORB_Det,
    AKAZE_Det,
    SIFT_Det
};

enum DescriptorType
{
    BRISK_Dsc,
    ORB_Dsc,
    FREAK_Dsc,
    AKAZE_Dsc,
    SIFT_Dsc
};


#endif /* dataStructures_h */
