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
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        // FLANN matching
        matcher = cv::FlannBasedMatcher::create();
        cout << "FLANN matching";
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        vector<vector<cv::DMatch>> knn_matches;
        double t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knn_matches, 2);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (KNN) with n=" << knn_matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;

        // filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {

            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
        cout << "# keypoints removed with KNN = " << knn_matches.size() - matches.size() << endl;
    }
    else{
        cout << "INVALID SELECTOR TYPE" << endl;
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, DescriptorType descriptorType,double &time)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor = nullptr;
    string descriptorTypeName;

    switch (descriptorType)
    {
    case BRIEF_Dsc:
    {
        int bytes = 32;        // Length of the descriptor in bytes, valid values are: 16, 32 (default) or 64
        bool use_orientation = false; // Sample patterns using keypoints orientation, disabled by default

        cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, use_orientation);
        descriptorTypeName = "BRIEF";
        break;
    }

    case ORB_Dsc:
    {
        int nfeatures = 500;                    // max number of features to retain
        float scaleFactor = 1.2f;               // pyramid decimation factor
        int nlevels = 8;                        // levels of pyramid
        int edgeThreshold = 31;                 // size of the border where features are not detected
        int firstLevel = 0;                     // the level of the pyramid to put the image
        int WTA_K = 2;                          // number of points used to produce oriented BRIEF descriptor
        cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;  // algorithm used to rank features
        int patchSize = 31;                     // size of patch used
        int fastThreshold = 20;                 // fast algorithm threshold

        extractor = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
        descriptorTypeName = "ORB";
        break;
    }

    case AKAZE_Dsc:
    {
        cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
        int descriptor_size = 0;
        int descriptor_channels = 3;
        float threshold = 0.001f;
        int nOctaves = 4;
        int nOctaveLayers = 4;
        cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;


        extractor = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity);
        descriptorTypeName = "AKAZE";
        break;
    }

    case FREAK_Dsc:
    {
        bool orientationNormalized = true;
        bool scaleNormalized = true;
        float patternScale = 22.0f;
        int nOctaves = 4;
        const std::vector<int> &selectedPairs = std::vector< int >();

        extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale, nOctaves, selectedPairs);
        descriptorTypeName = "FREAK";
        break;
    }

    case SIFT_Dsc:
    {
        int nfeatures = 0;
        int nOctaveLayers = 3;
        double contrastThreshold = 0.04;
        double edgeThreshold = 10;
        double sigma = 1.6;
        bool enable_precise_upscale = false;

        extractor = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma, enable_precise_upscale);
        descriptorTypeName = "SIFT";
        break;
    }

    default:
        cout << "Invalid descriptor type" << endl;
        break;
    }

    // perform feature description
    if(extractor != nullptr){
        double t = (double)cv::getTickCount();
        extractor->compute(img, keypoints, descriptors);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        time = 1000 * t / 1.0;
        cout << descriptorTypeName << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    }
}

// Detect keypoints in image using the traditional Shi-Thomasi detector_name
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time)
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
    time = 1000 * t / 1.0;
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}

void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &time)
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
    time = 1000 * t / 1.0;
    cout << "HARRIS detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, DetectorType detectorType, double &time, bool bVis)
{
    cv::Mat imgresized;
    cv::Ptr<cv::FeatureDetector> Detector = nullptr;
    string detector_name = "";

    switch (detectorType)
    {
    case SHITOMASI_Det:
        detKeypointsShiTomasi(keypoints,img, time);
        detector_name= "SHITOMASI";
        break;

    case HARRIS_Det:
        detKeypointsHarris(keypoints, img, time);
        detector_name= "HARRIS";
        break;

    case FAST_Det:
    {
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
        Detector = cv::FastFeatureDetector::create(35,true, type);
        detector_name= "FAST";
        break;
    }

    case BRISK_Det:
        Detector = cv::BRISK::create();
        detector_name= "BRISK";
        break;

    case ORB_Det:
        Detector = cv::ORB::create();
        detector_name= "ORB";
        break;

    case AKAZE_Det:
        Detector = cv::AKAZE::create();
        detector_name= "AKAZE";
        break;

    case SIFT_Det:
        Detector = cv::SIFT::create();
        detector_name= "SIFT";
        break;

    default:
        detector_name= "None";
        break;
    }

    if (Detector != nullptr){
        // detect
        double t = (double)cv::getTickCount();
        img.size();
        cv::resize(img, imgresized, cv::Size(), 0.25, 0.25);
        Detector->detect(imgresized,keypoints);

        //calculate time
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        time = 1000 * t / 1.0;
        cout << detector_name + " with n = " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    }

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detector_name + " Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

}