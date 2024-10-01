#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "utils.h"

const std::string IMG1 = "C:\\Users\\kamil\\Desktop\\Repos\\prymat_detection\\img\\1.jpeg";
const std::string IMG2 = "C:\\Users\\kamil\\Desktop\\Repos\\prymat_detection\\img\\2.jpeg";
const std::string IMG3 = "C:\\Users\\kamil\\Desktop\\Repos\\prymat_detection\\img\\3.jpeg";

// HSV minV 150 maxS 40 - 1. wersja
// teraz - minH 10, maxH 150

int main()
{
    // Load an image
    cv::Mat img = cv::imread(IMG3);

    // Scale the image down
    int scale = 30;
    auto scaled_img = scaleImage(img, scale);
    
    // Convert image from BGR to HSV
    auto hsv_img = convertToHSV(scaled_img);

    // Apply thresholding based on lower and upper margins
    std::vector<uchar> lower_margin = {10, 0, 0};
    std::vector<uchar> upper_margin = {150, 255, 255};
    auto thresholded_img = applyHSVThresholding(hsv_img, lower_margin, upper_margin);

    // Apply erosion to black pixels
    auto eroded_img = applyErosion(thresholded_img, 3, 0);

    // Apply dilation to black pixels
    auto dilated_img = applyDilation(eroded_img, 3, 0);

    // Find all ROIs
    std::vector<cv::Vec4i> rois = findROIs(dilated_img, 75, 50, hsv_img.cols / 2, hsv_img.rows / 2);
    std::cout << "ROIs found: " << rois.size() << std::endl;

    // Save image with all ROIs marked
    cv::imwrite("../detection_ROIs.png", showROIs(dilated_img, rois));

    // Test all ROIs and retain only those that meet the criteria
    std::vector<cv::Vec4i> final_rois = analyseROIs(dilated_img, rois);
    std::cout << "ROIs marked: " << final_rois.size() << std::endl;

    // Adjust ROI coordinates so they fit the original image
    adjustScaledValues(final_rois, scale);

    // Save image with confirmed ROIs marked
    saveDetectionResults(img, final_rois, "IMG1");

    return 0;
}