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


#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "SfM.hpp"

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <thread>
#include <chrono>

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────
static const std::string DETECTOR = "ORB";
static const std::string MATCHER  = "HAMMING";

// ─────────────────────────────────────────────────────────────────────────────
// Frame
//   landmark_map[keypoint_index] = index into global_points3D
//   This is the registration bookkeeping that makes resection correct.
// ─────────────────────────────────────────────────────────────────────────────
struct Frame {
    cv::Mat                    image;
    std::vector<cv::KeyPoint>  keypoints;
    cv::Mat                    descriptors;
    cv::Mat                    R = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat                    t = cv::Mat::zeros(3, 1, CV_64F);
    bool                       registered = false;
    std::map<int, int>         landmark_map;   // kp_idx -> global 3D index
};

// ─────────────────────────────────────────────────────────────────────────────
// Camera intrinsics
// ─────────────────────────────────────────────────────────────────────────────
cv::Mat buildCameraMatrix()
{
    return (cv::Mat_<double>(3, 3) <<
        2905.88,       0, 1416.0,
              0, 2905.88, 1064.0,
              0,       0,     1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Load images
// ─────────────────────────────────────────────────────────────────────────────
std::vector<Frame> load_image_sequence(const std::string& dir)
{
    std::vector<Frame> frames;
    namespace fs = std::filesystem;
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        std::cerr << "Bad directory: " << dir << "\n"; return frames;
    }
    std::vector<fs::path> paths;
    for (const auto& e : fs::directory_iterator(dir)) {
        auto ext = e.path().extension().string();
        if (ext==".png"||ext==".jpg"||ext==".jpeg"||ext==".bmp"||ext==".JPG")
            paths.push_back(e.path());
    }
    std::sort(paths.begin(), paths.end());
    for (const auto& p : paths) {
        Frame f;
        f.image = cv::imread(p.string(), cv::IMREAD_GRAYSCALE);
        if (f.image.empty()) { std::cerr << "Failed: " << p << "\n"; continue; }
        std::cout << "Loaded: " << p.filename() << "\n";
        frames.push_back(f);
    }
    if (frames.size() < 2) std::cerr << "Warning: fewer than 2 images.\n";
    return frames;
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature detection
// ─────────────────────────────────────────────────────────────────────────────
void keypoints_detection(const cv::Mat& image,
                         std::vector<cv::KeyPoint>& kp,
                         cv::Mat& desc,
                         const std::string& det = "ORB")
{
    if (det == "AKAZE")
        cv::AKAZE::create()->detectAndCompute(image, cv::noArray(), kp, desc);
    else if (det == "SIFT")
        cv::SIFT::create(2000)->detectAndCompute(image, cv::noArray(), kp, desc);
    else
        cv::ORB::create(2000)->detectAndCompute(image, cv::noArray(), kp, desc);
    std::cout << "[" << det << "] " << kp.size() << " keypoints.\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Matching (ratio test)
// ─────────────────────────────────────────────────────────────────────────────
std::vector<cv::DMatch> keypoint_matching(const cv::Mat& d1,
                                          const cv::Mat& d2,
                                          const std::string& method = "HAMMING")
{
    std::vector<std::vector<cv::DMatch>> knn;
    if (method == "FLANN") {
        cv::Mat f1, f2;
        if (d1.type()!=CV_32F) d1.convertTo(f1,CV_32F); else f1=d1;
        if (d2.type()!=CV_32F) d2.convertTo(f2,CV_32F); else f2=d2;
        cv::FlannBasedMatcher().knnMatch(f1, f2, knn, 2);
    } else {
        cv::BFMatcher(cv::NORM_HAMMING).knnMatch(d1, d2, knn, 2);
    }
    std::vector<cv::DMatch> good;
    for (const auto& m : knn)
        if (m.size()==2 && m[0].distance < 0.75f*m[1].distance)
            good.push_back(m[0]);
    std::cout << "Good matches: " << good.size() << "\n";
    return good;
}

// ─────────────────────────────────────────────────────────────────────────────
// Two-view initialisation
// ─────────────────────────────────────────────────────────────────────────────
bool initialise(Frame& f0, Frame& f1,
                const cv::Mat& K,
                std::vector<cv::Point3f>& global_pts)
{
    auto matches = keypoint_matching(f0.descriptors, f1.descriptors, MATCHER);
    if (matches.size() < 8) {
        std::cerr << "Init: too few matches.\n"; return false;
    }

    std::vector<cv::Point2f> pts0, pts1;
    std::vector<int> idx0, idx1;
    for (const auto& m : matches) {
        pts0.push_back(f0.keypoints[m.queryIdx].pt); idx0.push_back(m.queryIdx);
        pts1.push_back(f1.keypoints[m.trainIdx].pt); idx1.push_back(m.trainIdx);
    }

    // Optional: show matches
    cv::Mat vis;
    cv::drawMatches(f0.image, f0.keypoints, f1.image, f1.keypoints, matches, vis,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), {},
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("Init Matches", vis);
    cv::waitKey(500);
    cv::destroyAllWindows();

    cv::Mat E = SfM::estimateEssentialMatrix(pts0, pts1, K);
    cv::Mat R, t;
    std::vector<cv::Point3f> pts3D;
    SfM::recoverPoseAndTriangulate(E, pts0, pts1, pts3D, K, R, t);

    if (pts3D.empty()) { std::cerr << "Init: no 3D points.\n"; return false; }

    f0.R = cv::Mat::eye(3,3,CV_64F);
    f0.t = cv::Mat::zeros(3,1,CV_64F);
    f1.R = R.clone();
    f1.t = t.clone();
    f0.registered = f1.registered = true;

    // Populate landmark maps
    // pts3D[k] came from match k (idx0[k] <-> idx1[k])
    for (int k = 0; k < (int)pts3D.size(); ++k) {
        const auto& p = pts3D[k];
        if (!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z)
                ||std::abs(p.z)>100.f) continue;
        int gi = (int)global_pts.size();
        global_pts.push_back(p);
        f0.landmark_map[idx0[k]] = gi;
        f1.landmark_map[idx1[k]] = gi;
    }
    // f0.landmark_map = idx0;
    // f1.landmark_map = idx1;

    std::cout << "Init OK: " << global_pts.size() << " 3D points.\n";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Resection  –  register new_frame using already-registered frames
//
//   Step 1: collect 3D-2D correspondences via landmark_map lookup
//   Step 2: solvePnPRansac → pose of new_frame
//   Step 3: update new_frame.landmark_map with inlier points
//   Step 4: triangulate new tracks (not yet in map) to extend the map
// ─────────────────────────────────────────────────────────────────────────────
bool resection(Frame&                    new_frame,
               std::vector<Frame*>&      reg_frames,
               const cv::Mat&            K,
               std::vector<cv::Point3f>& global_pts)
{
    if (new_frame.descriptors.empty())
        keypoints_detection(new_frame.image, new_frame.keypoints,
                            new_frame.descriptors, DETECTOR);

    // ── Step 1: gather 3D-2D correspondences ─────────────────────────────────
    std::vector<cv::Point3f> corr3D;
    std::vector<cv::Point2f> corr2D;
    std::vector<int>         corr_new_kp;   // new-frame keypoint index per corr.
    std::vector<int>         corr_gi;       // global point index per corr.

    for (Frame* ref : reg_frames) {
        if (!ref->registered || ref->descriptors.empty()) continue;

        auto matches = keypoint_matching(ref->descriptors,
                                         new_frame.descriptors, MATCHER);
        for (const auto& m : matches) {
            auto it = ref->landmark_map.find(m.queryIdx);
            if (it == ref->landmark_map.end()) continue;

            int gi = it->second;
            if (gi < 0 || gi >= (int)global_pts.size()) continue;

            corr3D.push_back(global_pts[gi]);
            corr2D.push_back(new_frame.keypoints[m.trainIdx].pt);
            corr_new_kp.push_back(m.trainIdx);
            corr_gi.push_back(gi);
        }
    }

    std::cout << "Resection: " << corr3D.size() << " correspondences.\n";
    if ((int)corr3D.size() < 6) {
        std::cerr << "Resection: too few correspondences.\n"; return false;
    }

    // ── Step 2: PnP RANSAC ────────────────────────────────────────────────────
    cv::Mat dist = cv::Mat::zeros(4,1,CV_64F);
    cv::Mat rvec, tvec;
    std::vector<int> inliers;

    bool ok = cv::solvePnPRansac(
        corr3D, corr2D, K, dist, rvec, tvec,
        false, 200, 4.0f, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);

    if (!ok || (int)inliers.size() < 6) {
        std::cerr << "solvePnPRansac failed (inliers: " << inliers.size() << ").\n";
        return false;
    }

    cv::Rodrigues(rvec, new_frame.R);
    new_frame.t = tvec.clone();
    new_frame.registered = true;
    std::cout << "Resection OK – inliers: " << inliers.size()
              << " / " << corr3D.size() << "\n";

    // ── Step 3: register inlier landmarks into new_frame.landmark_map ─────────
    for (int idx : inliers)
        new_frame.landmark_map[corr_new_kp[idx]] = corr_gi[idx];

    // ── Step 4: triangulate new tracks ────────────────────────────────────────
    for (Frame* ref : reg_frames) {
        if (!ref->registered || ref->descriptors.empty()) continue;

        auto matches = keypoint_matching(ref->descriptors,
                                         new_frame.descriptors, MATCHER);

        std::vector<cv::Point2f> new_ref_pts, new_new_pts;
        std::vector<int>         new_ref_idx, new_new_idx;

        for (const auto& m : matches) {
            bool ref_known = ref->landmark_map.count(m.queryIdx)   > 0;
            bool new_known = new_frame.landmark_map.count(m.trainIdx) > 0;
            if (ref_known || new_known) continue;   // already in map

            new_ref_pts.push_back(ref->keypoints[m.queryIdx].pt);
            new_new_pts.push_back(new_frame.keypoints[m.trainIdx].pt);
            new_ref_idx.push_back(m.queryIdx);
            new_new_idx.push_back(m.trainIdx);
        }

        if ((int)new_ref_pts.size() < 8) continue;

        cv::Mat Rt_ref, Rt_new, P_ref, P_new;
        cv::hconcat(ref->R, ref->t, Rt_ref);
        cv::hconcat(new_frame.R, new_frame.t, Rt_new);
        P_ref = K * Rt_ref;
        P_new = K * Rt_new;

        cv::Mat pts4D;
        cv::triangulatePoints(P_ref, P_new, new_ref_pts, new_new_pts, pts4D);

        for (int k = 0; k < pts4D.cols; ++k) {
            float w = pts4D.at<float>(3,k);
            if (std::abs(w) < 1e-6f) continue;
            cv::Point3f p(pts4D.at<float>(0,k)/w,
                          pts4D.at<float>(1,k)/w,
                          pts4D.at<float>(2,k)/w);
            if (!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z)
                    ||std::abs(p.z)>100.f) continue;
            int gi = (int)global_pts.size();
            global_pts.push_back(p);
            ref->landmark_map[new_ref_idx[k]]       = gi;
            new_frame.landmark_map[new_new_idx[k]] = gi;
        }
    }

    std::cout << "Map size after resection: " << global_pts.size() << "\n";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
pcl::PointCloud<pcl::PointXYZ>::Ptr buildCloud(const std::vector<cv::Point3f>& pts)
{
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    for (const auto& p : pts)
        if (std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z)
                &&std::abs(p.z)<100.f)
            cloud->points.emplace_back(p.x,p.y,p.z);
    cloud->width  = (uint32_t)cloud->points.size();
    cloud->height = 1; cloud->is_dense = false;
    return cloud;
}

void visualise(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const std::string& title)
{
    auto v = pcl::make_shared<pcl::visualization::PCLVisualizer>(title);
    v->setBackgroundColor(0,0,0);
    v->addPointCloud<pcl::PointXYZ>(cloud,"cloud");
    v->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE,2,"cloud");
    v->addCoordinateSystem(1.0);
    v->initCameraParameters();
    while (!v->wasStopped()) {
        v->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    const std::string image_dir =
        "/home/chiroma/Documents/projects/openMVG_Build/"
        "ImageDataset_SceauxCastle/images/";

    cv::Mat K = buildCameraMatrix();
    std::cout << "K:\n" << K << "\n\n";

    std::vector<Frame> frames = load_image_sequence(image_dir);
    if (frames.size() < 2) { std::cerr << "Need >= 2 frames.\n"; return 1; }

    // Detect features on all frames upfront
    for (auto& f : frames)
        keypoints_detection(f.image, f.keypoints, f.descriptors, DETECTOR);

    std::vector<cv::Point3f> global_pts;

    // Two-view initialisation
    if (!initialise(frames[0], frames[1], K, global_pts)) {
        std::cerr << "Init failed.\n"; return 1;
    }

    // // Incremental registration
    // std::vector<Frame*> reg = { &frames[0], &frames[1] };

    // for (size_t i = 2; i < frames.size(); ++i) {
    //     std::cout << "\n=== Frame " << i << " ===\n";
    //     if (resection(frames[i], reg, K, global_pts))
    //         reg.push_back(&frames[i]);
    //     else
    //         std::cerr << "Frame " << i << " skipped.\n";
    // }

    // std::cout << "\nRegistered " << reg.size() << "/" << frames.size() << " frames.\n";
    // std::cout << "Total 3D points: " << global_pts.size() << "\n";

    // if (global_pts.empty()) { std::cerr << "Empty map.\n"; return 1; }

    // Outlier removal
    auto raw   = buildCloud(global_pts);
    auto clean = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(raw); sor.setMeanK(50); sor.setStddevMulThresh(1.0);
    sor.filter(*clean);
    std::cout << "After SOR: " << clean->points.size() << " points.\n";

    pcl::io::savePCDFileBinary("reconstruction.pcd", *clean);
    visualise(clean, "Multi-view Reconstruction");
    return 0;
}


