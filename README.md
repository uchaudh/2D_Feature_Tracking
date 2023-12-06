# 2D Feature Tracking

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best.
This project consists of four parts:

* Loading images, setting up data structures and putting everything into a ring buffer to optimize memory load.
* Integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed.
* Descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson.
* In the last part, once the code framework is complete, the various algorithms are tested in different combinations and compared with regard to some performance measures.

## Dependencies for Running Locally
1. cmake >= 2.8
 * All OSes: [click here for installation instructions](https://cmake.org/install/)

2. make >= 4.1 (Linux, Mac), 3.81 (Windows)
 * Linux: make is installed by default on most Linux distros
 * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
 * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)

3. OpenCV >= 4.1
 * All OSes: refer to the [official instructions](https://docs.opencv.org/master/df/d65/tutorial_table_of_content_introduction.html)
 * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors. If using [homebrew](https://brew.sh/): `$> brew install --build-from-source opencv` will install required dependencies and compile opencv with the `opencv_contrib` module by default (no need to set `-DOPENCV_ENABLE_NONFREE=ON` manually).
 * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)

4. gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using either [MinGW-w64](http://mingw-w64.org/doku.php/start) or [Microsoft's VCPKG, a C++ package manager](https://docs.microsoft.com/en-us/cpp/build/install-vcpkg?view=msvc-160&tabs=windows). VCPKG maintains its own binary distributions of OpenCV and many other packages. To see what packages are available, type `vcpkg search` at the command prompt. For example, once you've _VCPKG_ installed, you can install _OpenCV 4.1_ with the command:
```bash
c:\vcpkg> vcpkg install opencv4[nonfree,contrib]:x64-windows
```
Then, add *C:\vcpkg\installed\x64-windows\bin* and *C:\vcpkg\installed\x64-windows\debug\bin* to your user's _PATH_ variable. Also, set the _CMake Toolchain File_ to *c:\vcpkg\scripts\buildsystems\vcpkg.cmake*.


## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.

## IMPLEMENTATION

### Data Buffer Optimization - MP 1

Implemented a ring buffer where new elements added from one side push out old elements from the other side.

```c++
if(dataBuffer.size() > dataBufferSize){
    dataBuffer.erase(dataBuffer.begin());
}
 dataBuffer.push_back(frame);
```

### Keypoint Detection - MP 2

Implemented HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT and made them selectable by setting an enum type. I took the extra step to add an enumeration because string type cannot be used in switch case for setting the detector.

```C++
/*********** main *******************/
// extract 2D keypoints from current image
vector<cv::KeyPoint> keypoints;     // create empty feature list for current image
// DetectorType detectorType = HARRIS_Det;   //set the required detector type
// DetectorType detectorType = SHITOMASI_Det;
// DetectorType detectorType = FAST_Det;
// DetectorType detectorType = BRISK_Det;
// DetectorType detectorType = ORB_Det;
// DetectorType detectorType = AKAZE_Det;
DetectorType detectorType = SIFT_Det;


// get keypoints from the required detector
double timeDet = 0.0;
detKeypointsModern(keypoints, imgGray, detectorType, timeDet, bVis);
NKeypoints.push_back(keypoints.size());
detectorTime.push_back(timeDet);

/*********** matching2D_Student.cpp *******************/
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
```

### Keypoint Removal - MP 3

Remove all keypoints outside of a pre-defined rectangle and only use the keypoints within the rectangle for further processing.

```C++
// only keep keypoints on the preceding vehicle
bool bFocusOnVehicle = true;
cv::Rect vehicleRect(555, 180, 140, 150);
vector<cv::KeyPoint> precedingVehPoints;
if (bFocusOnVehicle)
{
    for(auto keypoint : keypoints){
        bool precedingVehPoint = vehicleRect.contains(keypoint.pt);
        if(precedingVehPoint)
            precedingVehPoints.push_back(keypoint);
    }

    keypoints = precedingVehPoints;
}
```

### Keypoint Descriptors - MP 4

Implement descriptors BRIEF, ORB, FREAK, AKAZE and SIFT and make them selectable by setting a string accordingly.

```C++
/*********** main *******************/    
cv::Mat descriptors;
// DescriptorType descriptorType = BRISK_Dsc; // set the required descriptor
// DescriptorType descriptorType = ORB_Dsc;
// DescriptorType descriptorType = AKAZE_Dsc;
// DescriptorType descriptorType = FREAK_Dsc;
DescriptorType descriptorType = SIFT_Dsc;

double timeDsc = 0.0;
descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType,timeDsc);

// push descriptors for current frame to end of data buffer
(dataBuffer.end() - 1)->descriptors = descriptors;
descriptorTime.push_back(timeDet+timeDsc);

cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

/*********** matching2D_Student.cpp *******************/
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, DescriptorType descriptorType,double &time)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor = nullptr;
    string descriptorTypeName;

    switch (descriptorType)
    {
    case BRISK_Dsc:
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
        descriptorTypeName = "BRISK";
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
```

### Descriptor Matching - MP 5

Implement FLANN matching as well as k-nearest neighbor selection. Both methods must be selectable using the respective strings in the main function.

```C++
/*********** main *******************/
vector<cv::DMatch> matches;
string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
string selectorType = "SEL_NN";       // SEL_NN, SEL_KNN

matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                 (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                 matches, descriptorType, matcherType, selectorType);

NMatches.push_back(matches.size());

// store matches in current data frame
(dataBuffer.end() - 1)->kptMatches = matches;

/*********** matching2D_Student.cpp *******************/
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


        cout << "# keypoints removed with KNN = " << knn_matches.size() - matches.size() << endl;
    }
    else{
        cout << "INVALID SELECTOR TYPE" << endl;
    }
}
```

### Descriptor Distance Ratio - MP 6

Use the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints.

```C++
// filter matches using descriptor distance ratio test
double minDescDistRatio = 0.8;
for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
{

    if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
    {
        matches.push_back((*it)[0]);
    }
}
```

## Performance Evaluation

For all the averages, I have implemented simple math after the main for loop. The total number of keypoints, detector time for each image, total number of matches for each loop and descriptor time for each loop are stored in respective vectors and used here. All the parameters are recorded in the tables below.

```C++
float kptsSum = 0.0f;
float avgKeypoints = 0.0f;
double timeSum = 0.0f;
double avgTime = 0.0f;
float matchesSum = 0;
float avgMatches = 0.0f;
float dscTimeSum = 0.0f;
float avgDscTime = 0.0f;
// calculate average number of keypoints and time for the selected detector
for(int i = 0; i < NKeypoints.size(); i++)
{
    kptsSum += NKeypoints[i];
    timeSum += detectorTime[i];
    matchesSum += NMatches[i];
    dscTimeSum += descriptorTime[i];
}

avgKeypoints = kptsSum/NKeypoints.size();
avgTime = timeSum/detectorTime.size();
avgMatches = matchesSum/NMatches.size();
avgDscTime = dscTimeSum/descriptorTime.size();

cout << "Average number of points = " << avgKeypoints << endl;
cout << "Average detector time = " << avgTime << endl;
cout << "Average matches = " <<avgMatches << endl;
cout << "Keypoint detection and descriptor extraction time = " << avgDscTime << endl;

return 0;
```



### Average number of keypoints - MP 7

| Detector   | Average number of keypoints | Average evaluation time |
| :--------- | --------------------------: | ----------------------: |
| Harris     |                   **173.7** |             **6.42622** |
| Shi-Tomasi |                  **1342.3** |             **18.4187** |
| FAST       |                  **1515.9** |                **1.55** |
| BRISK      |                      2711.6 |                  25.835 |
| ORB        |                         500 |                 33.9456 |
| AKAZE      |                        1343 |                 49.8699 |
| SIFT       |                       188.5 |                 12.0983 |

### Average number of matches - MP 8

| Detector\Descriptor | FREAK   | BRIEF      | ORB         | AKAZE   | SIFT    |
| ------------------- | ------- | ---------- | ----------- | ------- | ------- |
| Harris              | 49.5556 | 0.2083     | **49.5444** | X       | 50.1428 |
| Shi-Tomasi          | 50      | 0.0139     | **50.2111** | X       | 50.1428 |
| FAST                | 50      | 5.1021e-42 | **50.2111** | X       | 50.1428 |
| BRISK               | 50      | 0.25       | 50.2111     | X       | 50.1428 |
| ORB                 | 41.6667 | 0.2111     | 50.2111     | X       | 50.1428 |
| AKAZE               | 50      | 3.4037e-42 | 50.2111     | 50.0497 | 50.1428 |
| SIFT                | 34      | 0.1427     | X           | X       | 50.1428 |

### Average time for keypoint detection and descriptor extraction - MP 9

| Detector\Descriptor | FREAK   | BRIEF   | ORB        | AKAZE   | SIFT    |
| ------------------- | ------- | ------- | ---------- | ------- | ------- |
| Harris              | 23.9704 | 2.9239  | **3.8285** | X       | 14.4975 |
| Shi-Tomasi          | 25.0625 | 9.3949  | **8.7975** | X       | 18.9673 |
| FAST                | 19.1412 | 0.2517  | **1.4772** | X       | 12.3615 |
| BRISK               | 44.6435 | 4.5613  | 26.0654    | X       | 36.3256 |
| ORB                 | 32.3931 | 12.925  | 16.1103    | X       | 30.9962 |
| AKAZE               | 55.9926 | 3.56779 | 38.0481    | 66.4931 | 44.8852 |
| SIFT                | 22.8495 | 3.7817  | X          | X       | 61.5014 |

## Distribution of Neighborhood

- Harris, Shi-Tomasi and FAST have small neighborhood size and spacial distribution with no overlapping to each other.
- BRISK and ORB have very large neighborhood size and compact distribution like cluster with many overlapping with each other.
- AKAZE and SIFT have medium neighborhood size and relatively uniform distribution with small amount overlapping to each other.

## Observations

* SIFT detector + ORB do not work together and throws a OutOfMemory error.
* AKAZE detector only works with AKAZE descriptor.

## Top 3 recommended detector/descriptor combinations

The following three pairs are suggested according to the above metrics.

* Harris + ORB
* Shi-Tomasi + ORB
* FAST + ORB
