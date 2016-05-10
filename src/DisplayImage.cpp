#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <boost/thread/thread.hpp>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <string>
#include <cv.h>

using namespace cv;
using namespace std;
using namespace pcl;

const float squareSize = 2.3;

class intrinsicContainer {
public:
	Mat M1, M2, D1, D2;
	intrinsicContainer();

};

intrinsicContainer::intrinsicContainer() {
}

class extrinsicContainer {
public:
	Mat R, T, R1, R2, P1, P2, Q;
	Rect roi1, roi2;
	extrinsicContainer();
};

extrinsicContainer::extrinsicContainer() {
}

class parameterContainer {
public:
	int SADWindowSize;
	int numberOfDisparities;
	int preFilterSize;
	int preFilterCap;
	int minDisparity;
	int textureThreshold;
	int uniquenessRatio;
	int speckleWindowSize;
	int speckleRange;
	int disp12MaxDiff;
	parameterContainer();
};

parameterContainer::parameterContainer() {
	SADWindowSize = 9;
	numberOfDisparities = 16;
	preFilterSize = 0;
	preFilterCap = 31;
	minDisparity = 0;
	textureThreshold = 10;
	uniquenessRatio = 15;
	speckleWindowSize = 100;
	speckleRange = 32;
	disp12MaxDiff = 1;
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> visualize(
		pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, string windowName) {
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
			new pcl::visualization::PCLVisualizer(windowName));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(
			cloud);
	viewer->addPointCloud<PointXYZRGB>(cloud, rgb, "reconstruction");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	return viewer;
}

/*
 * @desc Creates a Point Cloud pointer using the inbuilt reprojectImageTo3D method
 * @param depthMap -  depth map that was created through stereo location
 * @param rgbMat - cv::Mat object containing the left frame for color introduction
 * @Param qMat - 4 x 4 cv::Mat object containing the relation between left and right camera
 * @return a pointer to the PointCloud created with the above 3 parameters
 */

pcl::PointCloud<pcl::PointXYZRGB>::Ptr reprojectedPointCloud(Mat depthMap,
		Mat rgbMat, Mat qMat) {

	Mat recons;

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud(
			new pcl::PointCloud<pcl::PointXYZRGB>());
	Mat xyz;
	reprojectImageTo3D(depthMap, recons, qMat, false, CV_32F);
	pointcloud->width = static_cast<uint32_t>(depthMap.cols);
	pointcloud->height = static_cast<uint32_t>(depthMap.rows);
	pointcloud->is_dense = false;

	pcl::PointXYZRGB point;

	for (int i = 0; i < depthMap.rows; i++) {
		uchar* rgb_ptr = rgbMat.ptr<uchar>(i);
		uchar* depth_ptr = depthMap.ptr<uchar>(i);

		for (int j = 0; j < depthMap.cols; j++) {

			uchar d = depth_ptr[j];
			if (d == 0)
				continue;
			Point3f p = recons.at<Point3f>(i, j);

			point.z = p.z;
			point.x = p.x;
			point.y = p.y;

			point.b = rgb_ptr[3 * j];
			point.g = rgb_ptr[3 * j + 1];
			point.r = rgb_ptr[3 * j + 2];

			pointcloud->points.push_back(point);
		}

	}

	return pointcloud;

}

Mat runStereoBM(intrinsicContainer intrinsics, extrinsicContainer extrinsics,
		parameterContainer params, Mat leftImage, Mat rightImage, int mode) {

	enum {
		STEREO_BM = 0,
		STEREO_SGBM = 1,
		STEREO_HH = 2,
		STEREO_VAR = 3,
		STEREO_3WAY = 4
	};
	Ptr<StereoBM> bm = StereoBM::create(16, 9);
	Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);

	Size img_size = leftImage.size();
	Mat Q, M1, D1, M2, D2;
	Rect roi1, roi2;

	M1 = intrinsics.M1;
	M2 = intrinsics.M2;
	D1 = intrinsics.D1;
	D2 = intrinsics.D2;

	Mat R, T, R1, P1, R2, P2;

	R = extrinsics.R;
	T = extrinsics.T;
	R1 = extrinsics.R1;
	P1 = extrinsics.P1;
	R2 = extrinsics.R2;
	P2 = extrinsics.P2;
	roi1 = extrinsics.roi1;
	roi2 = extrinsics.roi2;

	Mat map11, map12, map21, map22;
	initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
	initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

	Mat img1r, img2r;
	remap(leftImage, img1r, map11, map12, INTER_LINEAR);
	remap(rightImage, img2r, map21, map22, INTER_LINEAR);

	leftImage = img1r;
	rightImage = img2r;

	imshow("win1", leftImage);
	imshow("win2", rightImage);

	bm->setROI1(roi1);
	bm->setROI2(roi2);
	bm->setNumDisparities(params.numberOfDisparities);
	bm->setPreFilterCap(params.preFilterCap);
	bm->setBlockSize(params.SADWindowSize);
	bm->setMinDisparity(params.minDisparity);
	bm->setTextureThreshold(params.textureThreshold);
	bm->setUniquenessRatio(params.uniquenessRatio);
	bm->setSpeckleWindowSize(params.speckleWindowSize);
	bm->setSpeckleRange(params.speckleRange);

	int cn = leftImage.channels();

	sgbm->setP1(8 * cn * params.SADWindowSize * params.SADWindowSize);
	sgbm->setP2(32 * cn * params.SADWindowSize * params.SADWindowSize);
	sgbm->setNumDisparities(params.numberOfDisparities);
	sgbm->setPreFilterCap(params.preFilterCap);
	sgbm->setBlockSize(params.SADWindowSize);
	sgbm->setMinDisparity(params.minDisparity);
	sgbm->setUniquenessRatio(params.uniquenessRatio);
	sgbm->setSpeckleWindowSize(params.speckleWindowSize);
	sgbm->setSpeckleRange(params.speckleRange);

	if (mode == STEREO_HH) {
		sgbm->setMode(StereoSGBM::MODE_HH);
	} else if (mode == STEREO_SGBM) {
		sgbm->setMode(StereoSGBM::MODE_SGBM);
	} else if (mode == STEREO_3WAY) {
		sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);
	}

	Mat disp, disp8;

	if (mode == STEREO_BM) {
		cvtColor(leftImage, leftImage, CV_RGB2GRAY);
		cvtColor(rightImage, rightImage, CV_RGB2GRAY);
		bm->compute(leftImage, rightImage, disp);
	} else if (mode == STEREO_SGBM || mode == STEREO_HH
			|| mode == STEREO_3WAY) {
		sgbm->compute(leftImage, rightImage, disp);
	}
	if (mode != STEREO_VAR) {
		disp.convertTo(disp8, CV_8U, 255 / (params.numberOfDisparities * 16.));
	} else {
		disp.convertTo(disp8, CV_8U);
	}

	return disp8;
}

void stereoCalibrate(vector<vector<Point2f> > cornersLeft,
		vector<vector<Point2f> > cornersRight, Size boardSize, Size imageSize,
		intrinsicContainer *intrinsics, extrinsicContainer *extrinsics) {

	int nimages = cornersLeft.size();
	vector<vector<Point3f> > objectPoints;
	objectPoints.resize(nimages);

	for (int i = 0; i < nimages; i++) {
		for (int j = 0; j < boardSize.height; j++) {
			for (int k = 0; k < boardSize.width; k++) {
				objectPoints[i].push_back(
						Point3f(k * squareSize, j * squareSize, 0));
			}
		}
	}
	cout << "Running stereo calibration..." << endl;

	Mat cameraMatrix[2], distCoeffs[2];
	cameraMatrix[0] = initCameraMatrix2D(objectPoints, cornersLeft, imageSize,
			0);
	cameraMatrix[1] = initCameraMatrix2D(objectPoints, cornersRight, imageSize,
			0);
	Mat R, T, E, F;

	double rms = stereoCalibrate(objectPoints, cornersLeft, cornersRight,
			cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1],
			imageSize, R, T, E, F,
			CALIB_FIX_ASPECT_RATIO + CALIB_ZERO_TANGENT_DIST
					+ CALIB_USE_INTRINSIC_GUESS + CALIB_SAME_FOCAL_LENGTH
					+ CALIB_RATIONAL_MODEL + CALIB_FIX_K3 + CALIB_FIX_K4
					+ CALIB_FIX_K5,
			TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));
	cout << "Done with RMS error = " << rms << endl;
	// CALIBRATION QUALITY CHECK
	// because the output fundamental matrix implicitly
	// includes all the output information,
	// we can check the quality of calibration using the
	// epipolar geometry constraint: m2^t*F*m1=0
	double err = 0;
	int npoints = 0;
	vector<Vec3f> lines[2];
	for (int i = 0; i < nimages; i++) {
		int npt = (int) cornersLeft[i].size();
		Mat imgpt[2];
		imgpt[0] = Mat(cornersLeft[i]);
		undistortPoints(imgpt[0], imgpt[0], cameraMatrix[0], distCoeffs[0],
				Mat(), cameraMatrix[0]);
		computeCorrespondEpilines(imgpt[0], 1, F, lines[0]);

		imgpt[1] = Mat(cornersRight[i]);
		undistortPoints(imgpt[1], imgpt[1], cameraMatrix[1], distCoeffs[1],
				Mat(), cameraMatrix[1]);
		computeCorrespondEpilines(imgpt[1], 2, F, lines[1]);
		for (int j = 0; j < npt; j++) {
			double errij = fabs(
					cornersLeft[i][j].x * lines[1][j][0]
							+ cornersLeft[i][j].y * lines[1][j][1]
							+ lines[1][j][2])
					+ fabs(
							cornersRight[i][j].x * lines[0][j][0]
									+ cornersRight[i][j].y * lines[0][j][1]
									+ lines[0][j][2]);
			err += errij;
		}
		npoints += npt;
	}
	cout << "Average epipolar err = " << err / npoints << endl;

	// save intrinsic and extrinsic parameters

	intrinsics->M1 = cameraMatrix[0];
	intrinsics->M2 = cameraMatrix[1];
	intrinsics->D1 = distCoeffs[0];
	intrinsics->D2 = distCoeffs[1];
	Rect roi1, roi2;

	Mat R1, R2, P1, P2, Q;

	stereoRectify(cameraMatrix[0], distCoeffs[0], cameraMatrix[1],
			distCoeffs[1], imageSize, R, T, R1, R2, P1, P2, Q,
			CALIB_ZERO_DISPARITY, -1, imageSize, &roi1, &roi2);

	extrinsics->R = R;
	extrinsics->T = T;
	extrinsics->R1 = R1;
	extrinsics->R2 = R2;
	extrinsics->P1 = P1;
	extrinsics->P2 = P2;
	extrinsics->Q = Q;
	extrinsics->roi1 = roi1;
	extrinsics->roi2 = roi2;
}

int main(int, char**) {

	// open the cameras
	VideoCapture camleft(0);

	if (!camleft.isOpened())
		return -1;
	VideoCapture camright(1);
	for (int i = 1; i < 100; i++) {
		camright.open(i);
		if (camright.isOpened()) {
			break;
		}
	}

	//create two windows for both cameras, used throughout
	namedWindow("corners1", CV_WINDOW_AUTOSIZE);
	namedWindow("corners2", CV_WINDOW_AUTOSIZE);

	Mat framel, framer, depthMap;
//	Size boardSize(7, 7);
	Size boardSize(9, 19);
	bool foundl, foundr, isCalibrated = false;

	//create enum for running mode, along with int to store mode
	enum {
		STANDBY = 0, CALIBRATE = 1, STEREO_DEPTH = 2, POINTCLOUD = 3
	};
	int currentMode = STANDBY;

	int mode = 0;

	//create vector for corner points
	vector<vector<Point2f> > imagePointsL;
	vector<vector<Point2f> > imagePointsR;

	//create containers for calibration data
	extrinsicContainer mainExtrinsics;
	intrinsicContainer mainIntrinsics;

	//create container for matching parameters
	parameterContainer mainParams;
	pcl::visualization::CloudViewer viewer("Simple Cloud");
	for (;;) {
		camleft >> framel; // get a new frame from camera
		camright >> framer;

		imshow("corners1", framel);
		imshow("corners2", framer);

		char keyboardInput = (char) waitKey(50);

		//Set mode
		if (keyboardInput == 'c') {
			if (currentMode != CALIBRATE) {
				cout << "Starting Calibration..." << endl;
				cout << "Press s to save image and r to run calibration, "
						<< "w to write calibration data to YML files, "
						<< "l to load calibration data from YML files, "
						<< "and n to return to standby." << endl;
				currentMode = CALIBRATE;
			}
		} else if (keyboardInput == 'n') {
			if (currentMode != STANDBY) {
				cout << "Returning to standby mode..." << endl;
				cout
						<< "Press c to calibrate, m to create depth map, and p to view point cloud."
						<< endl;
				currentMode = STANDBY;
			}
		} else if (keyboardInput == 'm') {
			if (currentMode != STEREO_DEPTH && isCalibrated) {
				cout << "Starting stereo depth map creation..." << endl;
				currentMode = STEREO_DEPTH;
			}
		} else if (keyboardInput == 'p') {
			if (currentMode != POINTCLOUD) {
				cout << "Starting point cloud creation..." << endl;
				if (!isCalibrated) {
					cout
							<< "Error: System not calibrated. Please calibrate system by pressing c."
							<< endl;
				}
				currentMode = POINTCLOUD;
			}
		}

		if (currentMode == STANDBY) {
			imshow("corners1", framel);
			imshow("corners2", framer);

			destroyWindow("win1");
			destroyWindow("win2");
			destroyWindow("test");
		} else if (currentMode == CALIBRATE) {

			vector<Point2f> cornersL, cornersR;
			foundl = false;
			foundr = false;
			foundl = findChessboardCorners(framel, boardSize, cornersL,
					CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
			foundr = findChessboardCorners(framer, boardSize, cornersR,
					CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
			drawChessboardCorners(framel, boardSize, cornersL, foundl);
			drawChessboardCorners(framer, boardSize, cornersR, foundr);

			if (keyboardInput == 's') {
				cout << "Saving corners..." << endl;
				if (foundl && foundr) {
					/*
					 cornerSubPix(framel, cornersL, Size(11, 11), Size(-1, -1),
					 TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,
					 30, 0.01));

					 cornerSubPix(framer, cornersR, Size(11, 11), Size(-1, -1),
					 TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,
					 30, 0.01));
					 */
					imagePointsL.push_back(cornersL);
					imagePointsR.push_back(cornersR);
					cout << "Corners saved." << endl;
					cout << "Number of corner sets stored: "
							<< imagePointsL.size() << "." << endl;
				} else {
					cout << "Error: corners not found" << endl;
				}
				keyboardInput = 0;
			} else if (keyboardInput == 'r') {
				if (imagePointsL.size() < 6) {
					int required = 6 - imagePointsL.size();
					cout << "Not enough image pairs. Need " << required
							<< " more image pairs." << endl;
				} else {
					Size imageSize = framer.size();

					stereoCalibrate(imagePointsL, imagePointsR, boardSize,
							imageSize, &mainIntrinsics, &mainExtrinsics);

					isCalibrated = true;
					cout << "Calibration complete" << endl;
				}
			} else if (keyboardInput == 'w') {
				if (isCalibrated) {
					cout << "Saving calibration data to file..." << endl;
					FileStorage fs("intrinsics.yml", FileStorage::WRITE);
					if (fs.isOpened()) {
						fs << "M1" << mainIntrinsics.M1 << "D1"
								<< mainIntrinsics.D1 << "M2"
								<< mainIntrinsics.M2 << "D2"
								<< mainIntrinsics.D2;
						fs.release();
						fs.open("extrinsics.yml", FileStorage::WRITE);
						if (fs.isOpened()) {
							fs << "R" << mainExtrinsics.R << "T"
									<< mainExtrinsics.T << "R1"
									<< mainExtrinsics.R1 << "R2"
									<< mainExtrinsics.R2 << "P1"
									<< mainExtrinsics.P1 << "P2"
									<< mainExtrinsics.P2 << "Q"
									<< mainExtrinsics.Q << "roi1"
									<< mainExtrinsics.roi1 << "roi2"
									<< mainExtrinsics.roi2;
							fs.release();
						}
						cout << "Calibration data saved" << endl;
					}
				}
			} else if (keyboardInput == 'l') {
				cout << "Loading calibration data from file..." << endl;
				FileStorage fs("intrinsics.yml", FileStorage::READ);
				if (fs.isOpened()) {
					fs["M1"] >> mainIntrinsics.M1;
					fs["D1"] >> mainIntrinsics.D1;
					fs["M2"] >> mainIntrinsics.M2;
					fs["D2"] >> mainIntrinsics.D2;
					fs.release();
				} else {
					cout << "Could not load intrinsic parameters from file."
							<< endl;
					keyboardInput = 'c';
					continue;
				}
				fs.open("extrinsics.yml", FileStorage::READ);
				if (fs.isOpened()) {
					fs["R"] >> mainExtrinsics.R;
					fs["T"] >> mainExtrinsics.T;
					fs["R1"] >> mainExtrinsics.R1;
					fs["R2"] >> mainExtrinsics.R2;
					fs["P1"] >> mainExtrinsics.P1;
					fs["P2"] >> mainExtrinsics.P2;
					fs["Q"] >> mainExtrinsics.Q;
					fs["roi1"] >> mainExtrinsics.roi1;
					fs["roi2"] >> mainExtrinsics.roi2;
					fs.release();
				} else {
					cout << "Could not load extrinsic parameters from file."
							<< endl;
					keyboardInput = 'c';
					continue;
				}
				cout << "Calibration data loaded. Returning to standby."
						<< endl;
				isCalibrated = true;
				currentMode = STANDBY;

			}
		} else if (currentMode == STEREO_DEPTH) {

			namedWindow("win1", CV_WINDOW_AUTOSIZE);
			namedWindow("win2", CV_WINDOW_AUTOSIZE);
			namedWindow("depthmap", CV_WINDOW_AUTOSIZE);
			bool update = true;
			bool running = true;
			char keyboardInput2;
			int paramEditMode;
			while (running) {
				if (update) {
					depthMap = runStereoBM(mainIntrinsics, mainExtrinsics,
							mainParams, framel, framer, mode);
					imshow("depthmap", depthMap);
					update = false;
				}
				keyboardInput2 = (char) waitKey(0);
				switch (keyboardInput2) {
				case 'a':
					if (paramEditMode != 1) {
						paramEditMode = 1;
						cout
								<< "Modifying SADWindowSize. Use U and I to modify."
								<< endl;
					}
					break;
				case 's':
					if (paramEditMode != 2) {
						paramEditMode = 2;
						cout
								<< "Modifying numberOfDisparities. Use U and I to modify."
								<< endl;
					}
					break;
				case 'd':
					if (paramEditMode != 3) {
						paramEditMode = 3;
						cout
								<< "Modifying preFilterSize. Use U and I to modify."
								<< endl;
					}
					break;
				case 'f':
					if (paramEditMode != 4) {
						paramEditMode = 4;
						cout << "Modifying minDisparity. Use U and I to modify."
								<< endl;
					}
					break;
				case 'g':
					if (paramEditMode != 5) {
						paramEditMode = 5;
						cout
								<< "Modifying textureThreshold. Use U and I to modify."
								<< endl;
					}
					break;
				case 'h':
					if (paramEditMode != 6) {
						paramEditMode = 6;
						cout
								<< "Modifying uniquenessRatio. Use U and I to modify."
								<< endl;
					}
					break;
				case 'j':
					if (paramEditMode != 7) {
						paramEditMode = 7;
						cout
								<< "Modifying speckleWindowSize. Use U and I to modify."
								<< endl;
					}
					break;
				case 'k':
					if (paramEditMode != 8) {
						paramEditMode = 8;
						cout << "Modifying speckleRange. Use U and I to modify."
								<< endl;
					}
					break;
				case 'l':
					if (paramEditMode != 9) {
						paramEditMode = 0;
						cout
								<< "Modifying displ2MaxDiff. Use U and I to modify."
								<< endl;
					}
					break;
				case 'u':
					cout << "Decreasing Value..." << endl;
					switch (paramEditMode) {

					case 1:
						if (mainParams.SADWindowSize == 5u) {
							cout << "Already at minimum: "
									<< mainParams.SADWindowSize << endl;
						} else {
							mainParams.SADWindowSize -= 2;
							cout << "New value: " << mainParams.SADWindowSize
									<< endl;
						}
						break;
					case 2:
						if (mainParams.numberOfDisparities == 16) {
							cout << "Already at minimum: "
									<< mainParams.numberOfDisparities << endl;
						} else {
							mainParams.numberOfDisparities -= 16;
							cout << "New value: "
									<< mainParams.numberOfDisparities << endl;
						}
						break;
					case 3:
						mainParams.preFilterSize -= 1;
						cout << "New value: " << mainParams.preFilterSize
								<< endl;
						break;
					case 4:
						if (mainParams.numberOfDisparities == 1) {
							cout << "Already at minimum: "
									<< mainParams.minDisparity << endl;
						} else {
							mainParams.minDisparity -= 1;
							cout << "New value: " << mainParams.minDisparity
									<< endl;
						}
						break;
					case 5:
						if (mainParams.textureThreshold == 0) {
							cout << "Already at minimum: "
									<< mainParams.textureThreshold << endl;
						} else {
							mainParams.textureThreshold -= 1;
							cout << "New value: " << mainParams.textureThreshold
									<< endl;
						}
						break;
					case 6:
						if (mainParams.uniquenessRatio == 0) {
							cout << "Already at minimum: "
									<< mainParams.uniquenessRatio << endl;
						} else {
							mainParams.uniquenessRatio -= 1;
							cout << "New value: " << mainParams.uniquenessRatio
									<< endl;
						}
						break;
					case 7:
						mainParams.speckleWindowSize -= 1;
						cout << "New value: " << mainParams.speckleWindowSize
								<< endl;
						break;
					case 8:
						mainParams.speckleRange -= 1;
						cout << "New value: " << mainParams.speckleRange
								<< endl;
						break;
					case 9:
						mainParams.disp12MaxDiff -= 1;
						cout << "New value: " << mainParams.disp12MaxDiff
								<< endl;
					}
					update = true;
					break;
				case 'i':
					cout << "Increasing Value..." << endl;
					switch (paramEditMode) {
					case 1:
						mainParams.SADWindowSize += 2;
						cout << "New value: " << mainParams.SADWindowSize
								<< endl;
						break;
					case 2:
						mainParams.numberOfDisparities += 16;
						cout << "New value: " << mainParams.numberOfDisparities
								<< endl;
						break;
					case 3:
						mainParams.preFilterSize += 1;
						cout << "New value: " << mainParams.preFilterSize
								<< endl;
						break;
					case 4:
						mainParams.minDisparity += 1;
						cout << "New value: " << mainParams.minDisparity
								<< endl;
						break;
					case 5:
						mainParams.textureThreshold += 1;
						cout << "New value: " << mainParams.textureThreshold
								<< endl;
						break;
					case 6:
						mainParams.uniquenessRatio += 1;
						cout << "New value: " << mainParams.uniquenessRatio
								<< endl;
						break;
					case 7:
						mainParams.speckleWindowSize += 1;
						cout << "New value: " << mainParams.speckleWindowSize
								<< endl;
						break;
					case 8:
						mainParams.speckleRange += 1;
						cout << "New value: " << mainParams.speckleRange
								<< endl;
						break;
					case 9:
						mainParams.disp12MaxDiff += 1;
						cout << "New value: " << mainParams.disp12MaxDiff
								<< endl;
					}
					update = true;
					break;
				case 'r':
					cout << "Resetting all parameters..." << endl;
					mainParams = parameterContainer();
					update = true;
					break;
				case 'c':
					cout << "Changing to StereoBM" << endl;
					mode = 0;
					update = true;
					break;
				case 'v':
					cout << "Changing to StereoSGBM" << endl;
					mode = 1;
					update = true;
					break;
				case 'b':
					cout << "Changing to StereoSGBM_HH" << endl;
					mode = 2;
					update = true;
					break;
				case 'n':
					cout << "Changing to StereoSGBM_VAR" << endl;
					cout << "Buggy. Staying true" << endl;
					//mode = 3;
					update = true;
					break;
				case 'm':
					cout << "Changing to StereoSGBM_3Way" << endl;
					mode = 4;
					update = true;
					break;
				case 'q':
					cout << "Stopping stereo depth map creation..." << endl;
					cout << "Returning to standby..." << endl;
					running = false;
					currentMode = STANDBY;
					break;
				}
			}
			destroyWindow("win1");
			destroyWindow("win2");
			destroyWindow("depthmap");
			currentMode = STANDBY;

		} else if (currentMode == POINTCLOUD) {
			depthMap = runStereoBM(mainIntrinsics, mainExtrinsics, mainParams,
					framel, framer, mode);
			//RGB image Mat is framel;
			//use PCL function on depthMap, Q and framel;

//			imshow("test", depthMap);
			if (keyboardInput == 'n') {
				currentMode = STANDBY;
				destroyWindow("win1");
				destroyWindow("win2");
				destroyWindow("test");

			}
			viewer.showCloud(
					reprojectedPointCloud(depthMap, framel, mainExtrinsics.Q));
			imshow("test", depthMap);

		}

		if (keyboardInput == 'q') {
			break;
		}
	}
	return 0;
}
