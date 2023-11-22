#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // ...
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        // ...
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else
    {

        //...
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}

void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    double t = (double)cv::getTickCount();
    double max_overlap = 0.0;
    // getting local maxima
    // parse through the image rows
    for(size_t i = 0; i < dst_norm.rows; i++){
        // parse through the image coloumns
        for(size_t j = 0; j < dst_norm.cols; j++){
            // get one pixel
            int response = (int)dst_norm.at<float>(i,j);
            if(response > minResponse){
                cv::KeyPoint new_keypoint;
                new_keypoint.pt = cv::Point2f(j,i);
                new_keypoint.size = apertureSize * 2;
                new_keypoint.response = response;

                // NMS
                bool overlap_flag = false;
                for(auto it = keypoints.begin(); it != keypoints.end(); ++it){
                    double overlap = cv::KeyPoint::overlap(new_keypoint, *it);
                    if(overlap > max_overlap){
                        overlap_flag = true;
                        if(new_keypoint.response > (*it).response){
                            *it = new_keypoint;
                            break;
                        }
                    }
                }
                if(!overlap_flag){
                    keypoints.push_back(new_keypoint);
                }
            }
        }
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "HARRIS detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}

void detKeypointsFast(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
    cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
    cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(35,true, type);

    double t = (double)cv::getTickCount();
    detector->detect(img,keypoints);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "FAST with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}

void detKeypointsBrisk(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
    cv::Ptr<cv::FeatureDetector> brisk = cv::BRISK::create();

    double t = (double)cv::getTickCount();
    brisk->detect(img,keypoints);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "BRISK with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}

void detKeypointsOrb(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
    cv::Ptr<cv::FeatureDetector> orbDetector = cv::ORB::create();

    double t = (double)cv::getTickCount();
    orbDetector->detect(img,keypoints);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "ORB with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}

void detKeypointsAkaze(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
    cv::Ptr<cv::FeatureDetector> akazeDetector = cv::AKAZE::create();

    double t = (double)cv::getTickCount();
    akazeDetector->detect(img,keypoints);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "AKAZE with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}

void detKeypointsSift(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img)
{
    cv::Ptr<cv::FeatureDetector> siftDetector = cv::SIFT::create();

    double t = (double)cv::getTickCount();
    siftDetector->detect(img,keypoints);

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "SIFT with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, DetectorType detectorType, bool bVis)
{
    string window = "";

    switch (detectorType)
    {
    case SHITOMASI:
        detKeypointsShiTomasi(keypoints, img);
        window = "SHITOMASI";
        break;

    case HARRIS:
        detKeypointsHarris(keypoints, img);
        window = "HARRIS";
        break;

    case FAST:
        detKeypointsFast(keypoints,img);
        window = "FAST";
        break;

    case BRISK:
        detKeypointsBrisk(keypoints,img);
        window = "BRISK";
        break;

    case ORB:
        detKeypointsOrb(keypoints,img);
        window = "ORB";
        break;

    case AKAZE:
        detKeypointsAkaze(keypoints,img);
        window = "ORB";
        break;

    case SIFT:
        detKeypointsSift(keypoints,img);
        window = "SIFT";
        break;

    default:
        window = "None";
        break;
    }

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = window + " Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

}