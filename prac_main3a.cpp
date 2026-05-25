// #include <opencv2/opencv.hpp>
// #include <opencv2/features2d.hpp>
// #include "SfM.hpp"
// #include <iostream>
// #include <string>

// #include <pcl/visualization/cloud_viewer.h>
// #include <pcl/visualization/pcl_visualizer.h>
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
// #include <pcl/filters/statistical_outlier_removal.h>
// #include <pcl/surface/mls.h>
// #include <pcl/features/normal_3d.h>
// #include <pcl/surface/poisson.h>
// #include <pcl/io/pcd_io.h>
// #include <filesystem>
// #include <vector>
// #include <map>

// #include <thread>
// #include <chrono>

// // ─────────────────────────────────────────────────────────────────────────────
// // Frame structure to hold per-image data
// // ─────────────────────────────────────────────────────────────────────────────
// struct Frame {
//     cv::Mat image;
//     std::vector<cv::KeyPoint> keypoints;
//     cv::Mat descriptors;
//     cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
//     cv::Mat t = cv::Mat::zeros(3, 1, CV_64F);
// };

// struct Data {
//     std::vector<cv::Point3f>& points3D,
//     std::vector<cv::Point2f>& points2D,
//     cv::Mat image;
//     std::vector<cv::KeyPoint> keypoints;
//     cv::Mat descriptors;
// }

// // ─────────────────────────────────────────────────────────────────────────────
// // Camera intrinsics
// // ─────────────────────────────────────────────────────────────────────────────
// cv::Mat buildCameraMatrix() {
//     double fx = 2905.88;
//     double fy = 2905.88;
//     double cx = 1416.0;
//     double cy = 1064.0;

//     return (cv::Mat_<double>(3, 3) <<
//         fx,  0, cx,
//          0, fy, cy,
//          0,  0,  1);
// }

// // ─────────────────────────────────────────────────────────────────────────────
// // Load all images from a directory into Frame objects
// // ─────────────────────────────────────────────────────────────────────────────
// std::vector<Frame> load_image_sequence(const std::string& directory_path) {
//     std::vector<Frame> frames;
//     namespace fs = std::filesystem;

//     if (!fs::exists(directory_path) || !fs::is_directory(directory_path)) {
//         std::cerr << "Directory does not exist: " << directory_path << std::endl;
//         return frames;
//     }

//     // Collect and sort paths for deterministic ordering
//     std::vector<fs::path> paths;
//     for (const auto& entry : fs::directory_iterator(directory_path)) {
//         const auto ext = entry.path().extension().string();
//         if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" ||
//             ext == ".bmp" || ext == ".JPG") {
//             paths.push_back(entry.path());
//         }
//     }
//     std::sort(paths.begin(), paths.end());

//     for (const auto& p : paths) {
//         Frame frame;
//         frame.image = cv::imread(p.string(), cv::IMREAD_GRAYSCALE);
//         if (frame.image.empty()) {
//             std::cerr << "Failed to load: " << p << std::endl;
//             continue;
//         }
//         frames.push_back(frame);
//     }

//     if (frames.size() < 2) {
//         std::cerr << "Warning: fewer than 2 images found in " << directory_path << std::endl;
//     }

//     return frames;
// }

// // ─────────────────────────────────────────────────────────────────────────────
// // Feature detection
// //   feature_detector: "ORB" | "AKAZE" | "SIFT"
// // ─────────────────────────────────────────────────────────────────────────────
// void keypoints_detection(const cv::Mat& image,
//                          std::vector<cv::KeyPoint>& keypoints,
//                          cv::Mat& descriptors,
//                          const std::string& feature_detector = "ORB")
// {
//     if (feature_detector == "ORB") {
//         std::cout << "ORB feature detector" << std::endl;
//         cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
//         orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

//     } else if (feature_detector == "AKAZE") {
//         std::cout << "AKAZE feature detector" << std::endl;
//         cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
//         akaze->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

//     } else if (feature_detector == "SIFT") {
//         std::cout << "SIFT feature detector" << std::endl;
//         cv::Ptr<cv::SIFT> sift = cv::SIFT::create(2000);
//         sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

//     } else {
//         std::cerr << "Unknown feature detector: " << feature_detector
//                   << ". Defaulting to ORB." << std::endl;
//         cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
//         orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
//     }
// }

// // ─────────────────────────────────────────────────────────────────────────────
// // Keypoint matching
// //   matching_method: "HAMMING" (ORB/AKAZE binary)  |  "FLANN" (SIFT float)
// // Returns good matches after ratio-test / distance filter.
// // ─────────────────────────────────────────────────────────────────────────────
// std::vector<cv::DMatch> keypoint_matching(const cv::Mat& desc1,
//                                           const cv::Mat& desc2,
//                                           const std::string& matching_method = "HAMMING")
// {
//     std::vector<cv::DMatch> good_matches;

//     if (matching_method == "HAMMING") {
//         // Brute-force with Hamming distance – suited for binary descriptors (ORB, AKAZE)
//         cv::BFMatcher matcher(cv::NORM_HAMMING, /*crossCheck=*/false);
//         std::vector<std::vector<cv::DMatch>> knn_matches;
//         matcher.knnMatch(desc1, desc2, knn_matches, 2);

//         // Lowe's ratio test
//         const float ratio_thresh = 0.75f;
//         for (const auto& m : knn_matches) {
//             if (m.size() == 2 && m[0].distance < ratio_thresh * m[1].distance) {
//                 good_matches.push_back(m[0]);
//             }
//         }

//     } else if (matching_method == "FLANN") {
//         // FLANN-based matching – suited for float descriptors (SIFT, SURF)
//         // Convert to CV_32F if needed
//         cv::Mat d1 = desc1, d2 = desc2;
//         if (d1.type() != CV_32F) d1.convertTo(d1, CV_32F);
//         if (d2.type() != CV_32F) d2.convertTo(d2, CV_32F);

//         cv::FlannBasedMatcher flann_matcher;
//         std::vector<std::vector<cv::DMatch>> knn_matches;
//         flann_matcher.knnMatch(d1, d2, knn_matches, 2);

//         // Lowe's ratio test
//         const float ratio_thresh = 0.75f;
//         for (const auto& m : knn_matches) {
//             if (m.size() == 2 && m[0].distance < ratio_thresh * m[1].distance) {
//                 good_matches.push_back(m[0]);
//             }
//         }

//     } else {
//         std::cerr << "Unknown matching method: " << matching_method
//                   << ". Defaulting to HAMMING." << std::endl;
//         return keypoint_matching(desc1, desc2, "HAMMING");
//     }

//     std::cout << "Good matches found: " << good_matches.size() << std::endl;
//     return good_matches;
// }

// // ─────────────────────────────────────────────────────────────────────────────
// // Resection (camera pose estimation from 2D-3D correspondences)
// //   Given a set of 3D world points and their 2D image projections,
// //   recover R and t using PnP + RANSAC.
// // Returns true on success.
// // ─────────────────────────────────────────────────────────────────────────────
// bool resection(std::vector<Frame>frames, const std::vector<cv::Point3f>& points3D,
//                const std::vector<cv::Point2f>& points2D,
//                const cv::Mat& K,
//                cv::Mat& R_out,
//                cv::Mat& t_out)
// {
    
//     for (auto& frame : frames) {
//         keypoints_detection(frame.image, frame.keypoints, frame.descriptors, DETECTOR);
//     }

//     for (size_t i = 1; i < frames.size(); ++i) {
//         std::vector<cv::DMatch> matches = keypoint_matching(frames[0].descriptors, frames[i].descriptors, MATCHER);
//         std::vector<cv::Point2f> matched_points2D1;
//         std::vector<cv::Point2f> matched_points2D2;
//         for (const auto& m : matches) {
//             matched_points2D1.push_back(frames[0].keypoints[m.queryIdx].pt);
//             matched_points2D2.push_back(frames[i].keypoints[m.trainIdx].pt);
//         }
//     }
    
//     if (points3D.size() < 4 || points3D.size() != points2D.size()) {
//         std::cerr << "Resection: insufficient or mismatched point correspondences." << std::endl;
//         return false;
//     }

//     cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);  // assume undistorted
//     cv::Mat rvec, tvec;
//     std::vector<int> inliers;

//     bool ok = cv::solvePnPRansac(
//         points3D, points2D, K, dist_coeffs,
//         rvec, tvec,
//         /*useExtrinsicGuess=*/false,
//         /*iterationsCount=*/200,
//         /*reprojectionError=*/4.0f,
//         /*confidence=*/0.99,
//         inliers,
//         cv::SOLVEPNP_ITERATIVE);

//     if (!ok || inliers.size() < 6) {
//         std::cerr << "Resection failed (inliers: " << inliers.size() << ")." << std::endl;
//         return false;
//     }

//     cv::Rodrigues(rvec, R_out);
//     t_out = tvec.clone();

//     std::cout << "Resection OK – inliers: " << inliers.size()
//               << " / " << points3D.size() << std::endl;
//     return true;
// }

// // ─────────────────────────────────────────────────────────────────────────────
// // Two-view initialisation: detect, match, estimate pose, triangulate
// // ─────────────────────────────────────────────────────────────────────────────
// void initialise(const std::vector<Frame>& frame1, const std::vector<Frame>& frame2, const cv::Mat& K, , cv::Mat& R, cv::Mat& t, std::vector<cv::Point3f> points3D)
// {
//     // cv::Mat img1 = cv::imread(path1, cv::IMREAD_GRAYSCALE);
//     // cv::Mat img2 = cv::imread(path2, cv::IMREAD_GRAYSCALE);

//     // if (img1.empty() || img2.empty()) {
//     //     std::cerr << "Error: could not load images: " << path1
//     //               << "  " << path2 << std::endl;
//     //     return;
//     // }

//     Data data;
//     cv::Mat img1 = frame1.image;
//     cv::Mat img2 = frame2.image;

//     // ── Feature detection ────────────────────────────────────────────────────
//     std::vector<cv::KeyPoint> kp1, kp2;
//     cv::Mat desc1, desc2;
//     keypoints_detection(img1, kp1, desc1, "ORB");
//     keypoints_detection(img2, kp2, desc2, "ORB");

//     if (desc1.empty() || desc2.empty()) {
//         std::cerr << "Error: no descriptors computed." << std::endl;
//         return;
//     }

//     // ── Matching ─────────────────────────────────────────────────────────────
//     std::vector<cv::DMatch> good_matches = keypoint_matching(desc1, desc2, "HAMMING");
//     std::vector<cv::DMatch> index;

//     // ── Extract matched point coordinates ────────────────────────────────────
//     std::vector<cv::Point2f> pts1, pts2;
//     for (const auto& m : good_matches) {
//         pts1.push_back(kp1[m.queryIdx].pt);
//         pts2.push_back(kp2[m.trainIdx].pt);
//         index.push_back(m.queryIdx);
//     }

//     // ── Visualise matches ─────────────────────────────────────────────────────
//     cv::Mat img_matches;
//     cv::drawMatches(img1, kp1, img2, kp2, good_matches, img_matches,
//                     cv::Scalar::all(-1), cv::Scalar::all(-1), {},
//                     cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//     cv::imshow("Matched Keypoints", img_matches);
//     cv::waitKey(0);
//     cv::destroyAllWindows();

//     // ── Essential matrix & pose recovery ─────────────────────────────────────
//     cv::Mat E = SfM::estimateEssentialMatrix(pts1, pts2, K);

//     // cv::Mat R, t;
//     // std::vector<cv::Point3f> points3D;
//     SfM::recoverPoseAndTriangulate(E, pts1, pts2, points3D, K, R, t);

//     std::cout << "Recovered Rotation:\n"    << R << std::endl;
//     std::cout << "Recovered Translation:\n" << t << std::endl;
//     std::cout << "3D points triangulated: " << points3D.size() << std::endl;

//     data.points3D = points3D;
//     data.points2D = pts1;
//     data.image = img1;
//     data.keypoints = kp1;
//     data.descriptors = desc1;
//     data.index = index;

//     // ── Build PCL cloud ───────────────────────────────────────────────────────
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
//     for (const auto& p : points3D) {
//         if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)
//             && std::abs(p.z) < 100.0f) {
//             cloud->points.emplace_back(p.x, p.y, p.z);
//         }
//     }
//     cloud->width  = static_cast<uint32_t>(cloud->points.size());
//     cloud->height = 1;
//     cloud->is_dense = false;

//     std::cout << "Point cloud size: " << cloud->points.size() << std::endl;

//     if (cloud->points.empty()) {
//         std::cerr << "Warning: point cloud is empty after filtering." << std::endl;
//         return;
//     }

//     // ── Visualise point cloud ─────────────────────────────────────────────────
//     pcl::visualization::PCLVisualizer::Ptr viewer(
//         new pcl::visualization::PCLVisualizer("Two-view Reconstruction"));
//     viewer->setBackgroundColor(0, 0, 0);
//     viewer->addPointCloud<pcl::PointXYZ>(cloud, "init_cloud");
//     viewer->setPointCloudRenderingProperties(
//         pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "init_cloud");
//     viewer->addCoordinateSystem(1.0);
//     viewer->initCameraParameters();

//     while (!viewer->wasStopped()) {
//         viewer->spinOnce(100);
//         std::this_thread::sleep_for(std::chrono::milliseconds(100));
//     }
// }

// // ─────────────────────────────────────────────────────────────────────────────
// // Main – multi-view SfM pipeline
// // ─────────────────────────────────────────────────────────────────────────────
// int main()
// {
//     const std::string image_dir =
//         "/home/chiroma/Documents/projects/openMVG_Build/ImageDataset_SceauxCastle/images/";

//     cv::Mat K = buildCameraMatrix();
//     std::cout << "Camera matrix K:\n" << K << "\n\n";

//     cv::Mat R, t;
//     std::vector<cv::Point3f> points3D;

//     // ── Load images ───────────────────────────────────────────────────────────
//     std::vector<Frame> frames = load_image_sequence(image_dir);
//     if (frames.size() < 2) {
//         std::cerr << "Not enough frames to reconstruct." << std::endl;
//         return 1;
//     }

//     // ── Two-view initialisation (first two frames) ────────────────────────────
//     // We pass full paths, so re-derive them from loaded frames would require
//     // storing paths in Frame.  For simplicity we run initialise separately:
//     // initialise(image_dir + "frame0.jpg", image_dir + "frame1.jpg", K);

//     initilise(std::vector<Frame>{frames[0], frames[1]}, K);

//     // ── Feature detection on all frames ──────────────────────────────────────
//     const std::string DETECTOR = "ORB";
//     const std::string MATCHER  = "HAMMING";

//     for (auto& frame : frames) {
//         keypoints_detection(frame.image, frame.keypoints, frame.descriptors, DETECTOR);
//     }



//     resection(std::vector<Frame>frames );
//     // // ── Multi-view matching & triangulation ───────────────────────────────────
//     // pcl::PointCloud<pcl::PointXYZ>::Ptr global_cloud(
//     //     new pcl::PointCloud<pcl::PointXYZ>);

//     // for (size_t i = 0; i < frames.size(); ++i) {
//     //     for (size_t j = i + 1; j < frames.size(); ++j) {

//     //         std::vector<cv::DMatch> good_matches =
//     //             keypoint_matching(frames[i].descriptors,
//     //                               frames[j].descriptors,
//     //                               MATCHER);

//     //         if (good_matches.size() < 8) {
//     //             std::cout << "Skipping pair (" << i << ", " << j
//     //                       << "): too few matches." << std::endl;
//     //             continue;
//     //         }

//     //         // Extract matched point coordinates
//     //         std::vector<cv::Point2f> pts1, pts2;
//     //         for (const auto& m : good_matches) {
//     //             pts1.push_back(frames[i].keypoints[m.queryIdx].pt);
//     //             pts2.push_back(frames[j].keypoints[m.trainIdx].pt);
//     //         }

//     //         // Essential matrix & relative pose
//     //         cv::Mat E = SfM::estimateEssentialMatrix(pts1, pts2, K);
//     //         std::vector<cv::Point3f> points3D;
//     //         cv::Mat R_rel, t_rel;
//     //         SfM::recoverPoseAndTriangulate(E, pts1, pts2, points3D, K, R_rel, t_rel);

//     //         // Accumulate global pose for frame[j]
//     //         frames[j].R = R_rel * frames[i].R;
//     //         frames[j].t = R_rel * frames[i].t + t_rel;

//     //         // Add valid points to global cloud
//     //         for (const auto& p : points3D) {
//     //             if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)
//     //                 && std::abs(p.z) < 100.0f) {
//     //                 global_cloud->points.emplace_back(p.x, p.y, p.z);
//     //             }
//     //         }

//     //         std::cout << "Pair (" << i << ", " << j << "): "
//     //                   << points3D.size() << " 3D points  |  cloud total: "
//     //                   << global_cloud->points.size() << std::endl;
//     //     }
//     // }

//     // ── Finalise cloud ────────────────────────────────────────────────────────
//     global_cloud->width  = static_cast<uint32_t>(global_cloud->points.size());
//     global_cloud->height = 1;
//     global_cloud->is_dense = false;

//     std::cout << "\nTotal reconstructed points: "
//               << global_cloud->points.size() << std::endl;

//     if (global_cloud->points.empty()) {
//         std::cerr << "Global cloud is empty – nothing to visualise." << std::endl;
//         return 1;
//     }

//     // ── Statistical outlier removal ───────────────────────────────────────────
//     pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(
//         new pcl::PointCloud<pcl::PointXYZ>);
//     pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
//     sor.setInputCloud(global_cloud);
//     sor.setMeanK(50);
//     sor.setStddevMulThresh(1.0);
//     sor.filter(*filtered_cloud);

//     std::cout << "After outlier removal: "
//               << filtered_cloud->points.size() << " points" << std::endl;

//     // ── Visualise ─────────────────────────────────────────────────────────────
//     pcl::visualization::PCLVisualizer::Ptr viewer(
//         new pcl::visualization::PCLVisualizer("Multi-view Reconstruction"));
//     viewer->setBackgroundColor(0, 0, 0);
//     viewer->addPointCloud<pcl::PointXYZ>(filtered_cloud, "reconstruction");
//     viewer->setPointCloudRenderingProperties(
//         pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "reconstruction");
//     viewer->addCoordinateSystem(1.0);
//     viewer->initCameraParameters();

//     while (!viewer->wasStopped()) {
//         viewer->spinOnce(100);
//         std::this_thread::sleep_for(std::chrono::milliseconds(100));
//     }

//     return 0;
// }