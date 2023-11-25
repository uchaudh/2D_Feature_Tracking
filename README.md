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

## Performance Evaluation

### Average number of keypoints - MP 7

| Detector   | Average number of keypoints | Average evaluation time |
| :--------- | --------------------------: | ----------------------: |
| Harris     |                       173.7 |                 6.42622 |
| Shi-Tomasi |                      1342.3 |                 18.4187 |
| FAST       |                      1515.9 |                    1.55 |
| BRISK      |                      2711.6 |                  25.835 |
| ORB        |                         500 |                 33.9456 |
| AKAZE      |                        1343 |                 49.8699 |
| SIFT       |                       188.5 |                 12.0983 |

### Average number of matches MP8

| Detector\Descriptor | FREAK   | BRISK   | ORB     | AKAZE   | SIFT    |
| ------------------- | ------- | ------- | ------- | ------- | ------- |
| Harris              | 49.5556 | 49.8889 | 49.5444 | X       | 50.1428 |
| Shi-Tomasi          | 50      | 50      | 50.2111 | X       | 50.1428 |
| FAST                | 50      | 50      | 50.2111 | X       | 50.1428 |
| BRISK               | 50      | 50.3209 | 50.2111 | X       | 50.1428 |
| ORB                 | 41.6667 | 50.2111 | 50.2111 | X       | 50.1428 |
| AKAZE               | 50      | 50.0497 | 50.2111 | 50.0497 | 50.1428 |
| SIFT                | 34      | 39.8889 | X       | X       | 50.1428 |

### Average time for keypoint detection and descriptor extraction - MP9

| Detector\Descriptor | FREAK   | BRISK   | ORB     | AKAZE   | SIFT    |
| ------------------- | ------- | ------- | ------- | ------- | ------- |
| Harris              | 23.9704 | 7.15    | 3.8285  | X       | 14.4975 |
| Shi-Tomasi          | 25.0625 | 17.0252 | 8.7975  | X       | 18.9673 |
| FAST                | 19.1412 | 1.17123 | 1.4772  | X       | 12.3615 |
| BRISK               | 44.6435 | 26.3075 | 26.0654 | X       | 36.3256 |
| ORB                 | 32.3931 | 15.2501 | 16.1103 | X       | 30.9962 |
| AKAZE               | 55.9926 | 36.4504 | 38.0481 | 66.4931 | 44.8852 |
| SIFT                | 22.8495 | 10.3189 | X       | X       | 61.5014 |

## Observations

* SIFT detector + ORB descriptor do not work together
