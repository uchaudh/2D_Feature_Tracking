/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
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
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results
    vector<float> NKeypoints {};
    vector<double> detectorTime {};
    vector<double> descriptorTime {};
    vector<float> NMatches {};

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
        DataFrame frame;
        frame.cameraImg = imgGray;

        //// push image in a ring buffer
        if(dataBuffer.size() > dataBufferSize){
            dataBuffer.erase(dataBuffer.begin());
        }
         dataBuffer.push_back(frame);

        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints;     // create empty feature list for current image
        DetectorType detectorType = HARRIS_Det;   //set the required detector type
        // DetectorType detectorType = SHITOMASI_Det;
        // DetectorType detectorType = FAST_Det;
        // DetectorType detectorType = BRISK_Det;
        // DetectorType detectorType = ORB_Det;
        // DetectorType detectorType = AKAZE_Det;
        // DetectorType detectorType = SIFT_Det;


        // get keypoints from the required detector
        double timeDet = 0.0;
        detKeypointsModern(keypoints, imgGray, detectorType, timeDet, bVis);
        NKeypoints.push_back(keypoints.size());
        detectorTime.push_back(timeDet);

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

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = true;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << "NOTE: Keypoints have been limited to " << keypoints.size() << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        cv::Mat descriptors;
        DescriptorType descriptorType = BRIEF_Dsc; // set the required descriptor
        // DescriptorType descriptorType = ORB_Dsc;
        // DescriptorType descriptorType = AKAZE_Dsc;
        // DescriptorType descriptorType = FREAK_Dsc;
        // DescriptorType descriptorType = SIFT_Dsc;

        double timeDsc = 0.0;
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType,timeDsc);

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;
        descriptorTime.push_back(timeDet+timeDsc);

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

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

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    } // eof loop over all images

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
}