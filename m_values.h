#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

// BASE M VALUE + ~i and ~j

int convertValueTo01(int value)
{
    if (value == 255) return 0;
    else return 1;
}

double m(const cv::Mat& image, int p, int q) {
    CV_Assert(image.depth() != sizeof(uchar));
    double m = 0;
    switch (image.channels()) {
    case 1:
        for (int i = 0; i < image.rows; i++)
            for (int j = 0; j < image.cols; j++) {
                m += std::pow(i, p) * std::pow(j, q) * convertValueTo01(image.at<uchar>(i, j));
            }
        break;
    case 3:
        cv::Mat_<cv::Vec3b> _I = image;
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                m += std::pow(i, p) * std::pow(j, q) * convertValueTo01(_I(i, j)[0]);
            }
        }
        break;
    }

    return m;
}

double _i(const cv::Mat& image)
{
    double value = m(image, 1, 0) / m(image, 0, 0);

    return value;
}

double _j(const cv::Mat& image)
{
    double value = m(image, 0, 1) / m(image, 0, 0);

    return value;
}

// CENTRAL M VALUES

double m20(const cv::Mat& image)
{
    CV_Assert(image.depth() != sizeof(uchar));
    double value = (m(image, 2, 0) - static_cast<float>(std::pow(m(image, 1, 0), 2) / m(image, 0, 0)));

    return value;
}

double m02(const cv::Mat& image)
{
    CV_Assert(image.depth() != sizeof(uchar));
    double value = (m(image, 0, 2) - static_cast<float>(std::pow(m(image, 0, 1), 2) / m(image, 0, 0)));

    return value;
}

double m00(const cv::Mat& image)
{
    CV_Assert(image.depth() != sizeof(uchar));
    double value = m(image, 0, 0);

    return value;
}

double m11(const cv::Mat& image)
{
    CV_Assert(image.depth() != sizeof(uchar));
    double value = m(image, 1, 1) - m(image, 1, 0) * m(image, 0, 1) / m(image, 0, 0);

    return value;
}

double m30(const cv::Mat& image)
{
    CV_Assert(image.depth() != sizeof(uchar));
    double value = m(image, 3, 0) - 3 * m(image, 2, 0) * _i(image) + 2 * m(image, 1, 0) * pow(_i(image), 2);
    
    return value;
}

double m03(const cv::Mat& image)
{
    CV_Assert(image.depth() != sizeof(uchar));
    double value = m(image, 0, 3) - 3 * m(image, 0, 2) * _j(image) + 2 * m(image, 0, 1) * pow(_j(image), 2);
    
    return value;
}

double m12(const cv::Mat& image)
{
    CV_Assert(image.depth() != sizeof(uchar));
    double value = m(image, 1, 2) - 2 * m(image, 1, 1) * _j(image) - m(image, 0, 2) * _i(image) + 2 * m(image, 1, 0) * pow(_j(image), 2);
    
    return value;
}

double m21(const cv::Mat& image)
{
    CV_Assert(image.depth() != sizeof(uchar));
    double value = m(image, 2, 1) - 2 * m(image, 1, 1) * _i(image) - m(image, 2, 0) * _j(image) + 2 * m(image, 0, 1) * pow(_i(image), 2);
    
    return value;
}

// HU M VALUES

double getM1(const cv::Mat& image)
{
    // m20, m02 and m00
    CV_Assert(image.depth() != sizeof(uchar));
    double value = (m20(image) + m02(image)) / std::pow(m00(image), 2);

    return value;
}

double getM2(const cv::Mat& image)
{
    // m20, m02, m11 and m00
    CV_Assert(image.depth() != sizeof(uchar));
    double value = (pow(m20(image) - m02(image), 2) + 4.0 * pow(m11(image), 2)) / pow(m00(image), 4);

    return value;
}

double getM3(const cv::Mat& image)
{
    // m30, m12, m21, m03 and m00
    CV_Assert(image.depth() != sizeof(uchar));
    double value = (pow(m30(image) - 3 * m12(image), 2) + pow(3 * m21(image) - m03(image), 2)) / pow(m00(image), 5);

    return value;
}

double getM4(const cv::Mat& image)
{
    // m30, m12, m21, m03 and m00
    CV_Assert(image.depth() != sizeof(uchar));
    double value = (pow(m30(image) + m12(image), 2) + pow(m21(image) + m03(image), 2)) / pow(m00(image), 5);

    return value;
}

double getM5(const cv::Mat& image)
{
    // m30, m12, m21, m03 and m00
    CV_Assert(image.depth() != sizeof(uchar));
    double value = ((m30(image) - 3 * m12(image)) * (m30(image) + m12(image)) * (pow(m30(image) + m12(image), 2) - 3 * pow(m21(image) + m03(image), 2)) + (3 * m21(image) - m03(image)) * (m21(image) + m03(image)) * (3 * pow(m30(image) + m12(image), 2) - pow(m21(image) + m03(image), 2))) / pow(m00(image), 10);

    return value;
}

double getM6(const cv::Mat& image)
{
    // m20, m02, m30, m12, m21, m03, m11 and m00
    CV_Assert(image.depth() != sizeof(uchar));
    double value = ((m20(image) - m02(image)) * (pow(m30(image) + m12(image), 2) - pow(m21(image) + m03(image), 2)) + 4 * m11(image) * (m30(image) + m12(image)) * (m21(image) + m03(image))) / pow(m00(image), 7);

    return value;
}

double getM7(const cv::Mat& image)
{
    // m20, m02, m11 and m00
    CV_Assert(image.depth() != sizeof(uchar));
    double value = (m20(image) * m02(image) - std::pow(m11(image), 2)) / std::pow(m00(image), 4);

    return value;
}

double getM8(const cv::Mat& image)
{
    // m30, m12, m21, m03 and m00
    CV_Assert(image.depth() != sizeof(uchar));
    double value = (m30(image) * m12(image) + m21(image) * m03(image) - pow(m12(image), 2) - pow(m21(image), 2)) / pow(m00(image), 5);

    return value;
}

double getM9(const cv::Mat& image)
{
    // m20, m21, m03, m12, m02, m11, m30 and m00
    CV_Assert(image.depth() != sizeof(uchar));
    double value = (m20(image) * (m21(image) * m03(image) - pow(m12(image), 2)) + m02(image) * (m03(image) * m12(image) - pow(m21(image), 2)) - m11(image) * (m30(image) * m03(image) - m21(image) * m12(image))) / pow(m00(image), 7);

    return value;
}

double getM10(const cv::Mat& image)
{
    // m30, m03, m12, m21 and m00
    CV_Assert(image.depth() != sizeof(uchar));
    double value = (pow(m30(image) * m03(image) - m12(image) * m21(image), 2) - 4 * (m30(image) * m12(image) - pow(m21(image), 2)) * (m03(image) * m21(image) - m12(image))) / pow(m00(image), 10);

    return value;
}

int getArea(const cv::Mat& image, int value) {
    CV_Assert(image.depth() != sizeof(uchar));
    int area = 0;
    switch (image.channels())
    {
    case 1:
        for (int i = 0; i < image.rows; i++)
        {
            for (int j = 0; j < image.cols; j++)
            {
                if (image.at<uchar>(i, j) == value)
                {
                    area += 1;
                }
            }
        }
        break;
    case 3:
        cv::Mat_<cv::Vec3b> _I = image;
        for (int i = 0; i < image.rows; i++)
        {
            for (int j = 0; j < image.cols; j++)
            {
                if (_I(i, j)[0] == value)
                {
                    area += 1;
                }
            }
        }
        break;
    }

    return area;
}

int getPerimeter(const cv::Mat& image) {
    CV_Assert(image.depth() != sizeof(uchar));
    int perimeter = 0;
    switch (image.channels())
    {
    case 1:
        for (int i = 1; i < image.rows - 1; i++) {
            for (int j = 1; j < image.cols - 1; j++) {
                if (image.at<uchar>(i, j) == 0 && (image.at<uchar>(i - 1, j - 1) == 255 || image.at<uchar>(i, j - 1) == 255
                    || image.at<uchar>(i - 1, j) == 255 || image.at<uchar>(i + 1, j + 1) == 255
                    || image.at<uchar>(i, j + 1) == 255 || image.at<uchar>(i + 1, j) == 255
                    || image.at<uchar>(i - 1, j + 1) == 255 || image.at<uchar>(i + 1, j - 1) == 255))
                {
                    perimeter += 1;
                }
            }
        }
        break;
    case 3:
        cv::Mat_<cv::Vec3b> _I = image;
        for (int i = 1; i < image.rows - 1; i++) {
            for (int j = 1; j < image.cols - 1; j++) {
                if (_I(i, j)[0] == 0 && (_I(i - 1, j - 1)[0] == 255 || _I(i, j - 1)[0] == 255
                    || _I(i - 1, j)[0] == 255 || _I(i + 1, j + 1)[0] == 255
                    || _I(i, j + 1)[0] == 255 || _I(i + 1, j)[0] == 255
                    || _I(i - 1, j + 1)[0] == 255 || _I(i + 1, j - 1)[0] == 255))
                {
                    perimeter += 1;
                }
            }
        }
        break;
    }
    
    return perimeter;
}