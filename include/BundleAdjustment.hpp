#ifndef BUNDLE_ADJUSTMENT_HPP
#define BUNDLE_ADJUSTMENT_HPP

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/opencv.hpp>

struct ReprojectionError {
    ReprojectionError(double observed_x, double observed_y)
        : observed_x(observed_x), observed_y(observed_y) {}

    template <typename T>
    bool operator()(const T* const camera,
                   const T* const point,
                   T* residuals) const {
        // Camera parameters: rotation (3), translation (3), focal length (1)
        T p[3];
        // Apply rotation
        ceres::AngleAxisRotatePoint(camera, point, p);
        // Apply translation
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // Project to image plane
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        // Apply focal length
        T predicted_x = camera[6] * xp;
        T predicted_y = camera[6] * yp;

        // Compute residuals
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);

        return true;
    }

    static ceres::CostFunction* Create(double observed_x, double observed_y) {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 7, 3>(
            new ReprojectionError(observed_x, observed_y));
    }

    double observed_x;
    double observed_y;
};

class BundleAdjuster {
public:
    static void adjust(std::vector<cv::Point3f>& points3D,
                      const std::vector<cv::Point2f>& points1,
                      const std::vector<cv::Point2f>& points2,
                      cv::Mat& R,
                      cv::Mat& t);
};

#endif