#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"

void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time);
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time);
void detKeypointsFast(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time);
void detKeypointsBrisk(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time);
void detKeypointsOrb(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time);
void detKeypointsAkaze(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time);
void detKeypointsSift(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time);
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, DetectorType detectorType, double &time, bool bVis=false);
void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, DescriptorType descriptorType,double &time);
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType);

#endif /* matching2D_hpp */
