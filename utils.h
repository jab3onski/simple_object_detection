#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include <fstream>
#include <stack>
#include "m_values.h"

namespace fs = std::filesystem;

//! HELPER METHODS

// Make sure the conversion to uchar maintains the min and max value of 0 - 255
uchar correctColorRange(float pixel)
{
    if (pixel > 255) pixel = 255;
    else if (pixel < 0) pixel = 0;

    return uchar(pixel);
}

// Adjusts the values in ROI vector so they fit the original image
void adjustScaledValues(std::vector<cv::Vec4i>& rois, double scale) {

    double scale_factor = scale / 100.0;

    for (cv::Vec4i& roi : rois) {
        roi[0] = static_cast<int>(roi[0] / scale_factor);
        roi[1] = static_cast<int>(roi[1] / scale_factor);
        roi[2] = static_cast<int>(roi[2] / scale_factor);
        roi[3] = static_cast<int>(roi[3] / scale_factor);
    }
}

//! CORE METHODS

// Initiate flood fill algorithm to find boundaries of white pixel regions
void floodFillImage(const cv::Mat& image, cv::Mat& flood_image, int label, int x, int y, int& min_x, int& min_y, int& max_x, int& max_y)
{
    std::stack<std::pair<int, int>> stack;
    stack.push(std::make_pair(x, y));

    while (!stack.empty())
    {
        std::pair<int, int> p = stack.top();
        stack.pop();
        int px = p.first;
        int py = p.second;

        flood_image.at<int>(py, px) = label;

        min_x = std::min(min_x, px);
        min_y = std::min(min_y, py);
        max_x = std::max(max_x, px);
        max_y = std::max(max_y, py);

        // Check 8 neighbors
        for (int ny = -1; ny <= 1; ny++)
        {
            for (int nx = -1; nx <= 1; nx++)
            {
                if (ny == 0 && nx == 0) continue;
                int nxp = px + nx;
                int nyp = py + ny;
                
                if (nxp >= 0 && nxp < image.cols && nyp >= 0 && nyp < image.rows && image.at<uchar>(nyp, nxp) == 255 && flood_image.at<int>(nyp, nxp) == 0)
                {
                    stack.push(std::make_pair(nxp, nyp));
                }
            }
        }
    }
}

// Initiate flood fill algorithm to replace pixel clusters with given values
void floodFillImage(cv::Mat& image, int x, int y, uchar in_pixel_val, uchar out_pixel_val)
{
    std::stack<std::pair<int, int>> stack;
    stack.push(std::make_pair(x, y));

    if (image.at<uchar>(y, x) != 0) return;

    while (!stack.empty())
    {
        std::pair<int, int> p = stack.top();
        stack.pop();
        int px = p.first;
        int py = p.second;

        image.at<uchar>(py, px) = out_pixel_val;

        // Check 8 neighbors
        for (int ny = -1; ny <= 1; ny++)
        {
            for (int nx = -1; nx <= 1; nx++)
            {
                if (ny == 0 && nx == 0) continue;
                int nxp = px + nx;
                int nyp = py + ny;

                if (nxp >= 0 && nxp < image.cols && nyp >= 0 && nyp < image.rows && image.at<uchar>(nyp, nxp) == in_pixel_val)
                {
                    stack.push(std::make_pair(nxp, nyp));
                }
            }
        }
    }
}

// Initiate ROI search utilising flood fill algorithm to extract ROI box top left and bottom right coordinates
std::vector<cv::Vec4i> findROIs(const cv::Mat& image, int min_width, int min_height, int max_width, int max_height)
{
    std::vector<cv::Vec4i> rois;

    cv::Mat flood_image(image.size(), CV_32S, cv::Scalar(0));

    int label = 1;
    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            if (image.at<uchar>(y, x) == 255 && flood_image.at<int>(y, x) == 0)
            {
                int minX = x;
                int minY = y;
                int maxX = x;
                int maxY = y;
                floodFillImage(image, flood_image, label, x, y, minX, minY, maxX, maxY);
                
                if ((maxX - minX >= min_width) && (maxY - minY >= min_height) && (maxX - minX <= max_width) && (maxY - minY <= max_height))
                {
                    rois.push_back(cv::Vec4i(minX, minY, maxX, maxY));

                }

                label++;
            }
        }
    }

    return rois;
}

//! IMAGE MANIPULATION

// Draw a rectangle of given color using provided coordinates of top left and bottom right corners
void drawRectangle(cv::Mat& image, int x1, int y1, int x2, int y2, cv::Vec3b color)
{
    for (int x = x1; x <= x2; x++) 
    {
        image.at<cv::Vec3b>(y1, x) = color;
        image.at<cv::Vec3b>(y2, x) = color;
    }

    for (int y = y1; y <= y2; y++)
    {
        image.at<cv::Vec3b>(y, x1) = color;
        image.at<cv::Vec3b>(y, x2) = color;
    }
}

// Initiate edge cluster removal algorithm utilising flood fill to remove targeted clusters
cv::Mat removeClusters(const cv::Mat& image, uchar in_pixel_val, uchar out_pixel_val)
{
    int height = image.rows;
    int width = image.cols;

    cv::Mat out_image = image.clone();

    for (int x = 0; x < width; x++)
    {
        if (image.at<uchar>(0, x) == in_pixel_val) floodFillImage(out_image, x, 0, in_pixel_val, out_pixel_val);
        if (image.at<uchar>(height - 1, x) == in_pixel_val) floodFillImage(out_image, x, height - 1, in_pixel_val, out_pixel_val);
    }

    for (int y = 0; y < height; y++)
    {
        if (image.at<uchar>(y, 0) == in_pixel_val) floodFillImage(out_image, 0, y, in_pixel_val, out_pixel_val);
        if (image.at<uchar>(y, width - 1) == in_pixel_val) floodFillImage(out_image, width - 1, y, in_pixel_val, out_pixel_val);
    }

    return out_image;
}

// Initiaties dilation algorithm on given pixel values
cv::Mat applyDilation(const cv::Mat& image, int mask_size, uchar pixel_value)
{
    CV_Assert(mask_size >= 3 && mask_size % 2 == 1);
    cv::Mat out_image = image.clone();

    int height = image.rows;
    int width = image.cols;
    int radius = mask_size / 2;

    for (int y = radius; y < height - radius; y++)
    {
        for (int x = radius; x < width - radius; x++)
        {
            if (image.at<uchar>(y, x) == pixel_value)
            {
                for (int dy = -radius; dy <= radius; dy++)
                {
                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        out_image.at<uchar>(y + dy, x + dx) = image.at<uchar>(y, x);
                    }
                }
            }
        }
    }

    return out_image;
}

// Initiaties erosion algorithm on given pixel values
cv::Mat applyErosion(const cv::Mat& image, int maskSize, uchar pixel_value)
{
    CV_Assert(maskSize >= 3 && maskSize % 2 == 1);
    cv::Mat dst = image.clone();

    uchar opposite_pixel;
    if (pixel_value == 0) opposite_pixel = 255;
    else opposite_pixel = 0;

    int height = image.rows;
    int width = image.cols;
    int radius = maskSize / 2;

    for (int y = radius; y < height - radius; y++)
    {
        for (int x = radius; x < width - radius; x++)
        {
            if (image.at<uchar>(y, x) == pixel_value)
            {
                bool erode = false;
                for (int dy = -radius; dy <= radius; dy++)
                {
                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        if (image.at<uchar>(y + dy, x + dx) == opposite_pixel)
                        {
                            erode = true;
                            break;
                        }
                    }
                    if (erode) break;
                }
                if (erode) dst.at<uchar>(y, x) = opposite_pixel;
            }
        }
    }

    return dst;
}

// Initiaties thresholding algorithm based on lower and upper HSV values margins
cv::Mat applyHSVThresholding(const cv::Mat& image, std::vector<uchar> lower_margin, std::vector<uchar> upper_margin)
{
    // MARGINS
    // LOWER : 0, 0, 0
    // UPPER : 179, 255, 255
    cv::Mat out_img(image.rows, image.cols, CV_8U);

    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            uchar hue = image.at<cv::Vec3b>(y, x)[0];
            uchar saturation = image.at<cv::Vec3b>(y, x)[1];
            uchar value = image.at<cv::Vec3b>(y, x)[2];

            if (hue >= lower_margin.at(0) && hue <= upper_margin.at(0) &&
                saturation >= lower_margin[1] && saturation <= upper_margin[1] &&
                value >= lower_margin[2] && value <= upper_margin[2])
            {
                out_img.at<uchar>(y, x) = 255;
            }
            else out_img.at<uchar>(y, x) = 0;
        }
    }

    return out_img;
}

// Converts given BGR image to HSV palette
cv::Mat convertToHSV(const cv::Mat& image)
{
    cv::Mat out_img = cv::Mat(image.rows, image.cols, CV_8UC3);

    for (int y = 0; y < image.rows; y++) 
    {
        for (int x = 0; x < image.cols; x++)
        {
            double red = image.at<cv::Vec3b>(y, x)[2] / 255.0;
            double green = image.at<cv::Vec3b>(y, x)[1] / 255.0;
            double blue = image.at<cv::Vec3b>(y, x)[0] / 255.0;

            double Cmax = std::max({red, green, blue});
            double Cmin = std::min({red, green, blue});
            double delta = Cmax - Cmin;

            double value = Cmax;
            double hue = 0;
            double saturation;

            if (Cmax == red) hue = 60.0 * fmod((green - blue) / delta, 6);
            else if (Cmax == green) hue = 60.0 * (((blue - red) / delta) + 2);
            else if (Cmax == blue) hue = 60.0 * (((blue - red) / delta) + 4);
            if (hue < 0) hue += 360;

            if (Cmax == 0) saturation = 0.0;
            else saturation = delta / Cmax;

            out_img.at<cv::Vec3b>(y, x)[0] = static_cast<uchar>(hue / 2);
            out_img.at<cv::Vec3b>(y, x)[1] = static_cast<uchar>(saturation * 255);
            out_img.at<cv::Vec3b>(y, x)[2] = static_cast<uchar>(value * 255);
        }
    }

    return out_img;
}

// Scales the image down to given scaling factor (0.0+ - 1.0)
cv::Mat scaleImage(cv::Mat& image, double scale)
{
    int width = image.cols;
    int height = image.rows;
    int out_width = static_cast<int>(width * scale / 100.0);
    int out_height = static_cast<int>(height * scale / 100.0);

    cv::Mat out_image(out_height, out_width, image.type());

    double x_scale = static_cast<double>(width) / out_width;
    double y_scale = static_cast<double>(height) / out_height;

    for (int y = 0; y < out_height; y++)
    {
        for (int x = 0; x < out_width; x++)
        {
            int image_x = static_cast<int>(x * x_scale);
            int image_y = static_cast<int>(y * y_scale);

            out_image.at<cv::Vec3b>(y, x) = image.at<cv::Vec3b>(image_y, image_x);
        }
    }

    return out_image;
}

// Converts BGR image to grayscale
cv::Mat convertToGrayscale(cv::Mat image)
{
    CV_Assert(image.depth() != sizeof(uchar));
    cv::Mat_<cv::Vec3b> _I = image;
    cv::Mat grayscale_img(image.rows, image.cols, CV_8UC1, cv::Scalar(0));

    for (int y = 0; y < _I.rows; ++y) {
        for (int x = 0; x < _I.cols; ++x) {
            double intensity =
                (0.0722 * _I(y, x)[0]
                    + 0.7152 * _I(y, x)[1]
                    + 0.2126 * _I(y, x)[2]);
            grayscale_img.at<uchar>(y, x) = static_cast<uchar>(intensity);
        }
    }

    return grayscale_img;
}

// Initiaties thresholding algorithm based on given intensity threshold
cv::Mat applyGrayscaleThresholding(cv::Mat& image, int threshold)
{
    cv::Mat out_img(image.rows, image.cols, CV_8U);

    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            if (image.at<uchar>(y, x) > threshold) out_img.at<uchar>(y, x) = 255;
            else out_img.at<uchar>(y, x) = 0;
        }
    }

    return out_img;
}

//! M's AND OTHER ANALYSIS

// Checks a vector containing ROI coordinates and returns only those that pass tests
std::vector<cv::Vec4i> analyseROIs(cv::Mat& image, std::vector<cv::Vec4i> rois)
{
    std::vector<cv::Vec4i> confirmed_rois;
    for(auto& roi : rois)
    {
        int x1 = roi[0];
        int y1 = roi[1];
        int x2 = roi[2];
        int y2 = roi[3];

        auto roi_image_region = image(cv::Rect(x1, y1, x2 - x1, y2 - y1));
        cv::Mat corrected_roi_region = removeClusters(roi_image_region, 0, 255);

        double M6 = getM6(corrected_roi_region);
        double M6_dev = 0.001;
        double M6_average = 0.000384396;

        double M7 = getM7(corrected_roi_region);
        double M7_dev = 0.003;
        double M7_average = 0.022796325;

        double area_white = getArea(corrected_roi_region, 255);
        double area_black = getArea(corrected_roi_region, 0);
        double area_diff;
        if (area_black != 0) area_diff = area_white / area_black;
        else area_diff = 0;

        cv::Vec2d area_diff_average = cv::Vec2d(3, 5);
        cv::Vec2d M6_range = cv::Vec2d(M6_average - M6_dev, M6_average + M6_dev);
        cv::Vec2d M7_range = cv::Vec2d(M7_average - M7_dev, M7_average + M7_dev);

        if((M6 > M6_range[0]) && (M6 < M6_range[1]))
        {
            if ((M7 > M7_range[0]) && (M7 < M7_range[1]))
            {
                if ((area_diff > area_diff_average[0]) && (area_diff < area_diff_average[1]))
                {
                    confirmed_rois.push_back(roi);
                }
            }
        }
    }

    return confirmed_rois;
}

// Returns an image with all ROIs marked
cv::Mat showROIs(const cv::Mat& image, std::vector<cv::Vec4i> rois)
{
    cv::Mat out_image(image.rows, image.cols, CV_8UC3);

    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            uchar value = image.at<uchar>(y, x);
            out_image.at<cv::Vec3b>(y, x) = cv::Vec3b(value, value, value);
        }
    }

    int tag = 0;
    for(auto& roi : rois)
    {
        int x1 = roi[0];
        int y1 = roi[1];
        int x2 = roi[2];
        int y2 = roi[3];

        drawRectangle(out_image, x1, y1, x2, y2, cv::Vec3b(0, 0, 255));
        //auto cut_image = image(cv::Rect(x1, y1, x2 - x1, y2 - y1));
        //cv::Mat corrected_image = removeBlackClusters(cut_image);
    }

    return out_image;
}

// Saves an image with confirmed ROIs marked
void saveDetectionResults(const cv::Mat& image, std::vector<cv::Vec4i> rois, std::string name)
{
    cv::Mat out_image = image.clone();

    for(auto& roi : rois)
    {
        int x1 = roi[0];
        int y1 = roi[1];
        int x2 = roi[2];
        int y2 = roi[3];

        drawRectangle(out_image, x1, y1, x2, y2, cv::Vec3b(0, 255, 0));
        std::cout<<x1<<" "<<y1<<" "<<x2<<" "<<y2<<" "<<"\n";
    }

    cv::imwrite("../detection_" + name + ".jpeg", out_image);
}