#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <math.h>

std::vector<std::thread> threads;

using namespace std;

const int histSize = 256;

bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
    double i = fabs(contourArea(cv::Mat(contour1)));
    double j = fabs(contourArea(cv::Mat(contour2)));
    return (i < j);
}

uchar** getMatrix(cv::Mat mat) {
    uchar** array = new uchar* [mat.rows];
    for (int i = 0; i < mat.rows; i++)
        array[i] = new uchar[mat.cols * mat.channels()];

    for (int i = 0; i < mat.rows; i++)
        array[i] = mat.ptr<uchar>(i);

    return array;
}

cv::Mat getMat(uchar** matrix, int width, int height) {
    return cv::Mat(height, width, CV_8UC1, *matrix);
}

cv::Mat drawHistogram(cv::Mat& mat) {

    float range[] = { 0, 256 };
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;
    cv::Mat hist;

    int hist_w = 256;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);

    cv::Mat histImage(hist_h, hist_w, CV_8UC1, cv::Scalar(0));
    
    calcHist(&mat, 1, 0, cv::Mat(), hist, 1, &histSize, histRange, uniform, accumulate);
    normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

    for (int i = 1; i < histSize; i++) {
        cv::line(
            histImage,
            cv::Point(bin_w * (i - 1), cvRound(hist.at<float>(i - 1))),
            cv::Point(bin_w * (i), cvRound(hist.at<float>(i))),
            cv::Scalar(255, 0, 0), 2, 8, 0);
    }

    return histImage;
}

uchar getWhite(cv::Mat& hist) {

    vector<uchar> output;

    int max(0);
    int max_pos(0);
    bool onHill = false;

    for (int i = 255; i > 128 / 2; i--) {
        int current = hist.at<uchar>(i);
        if (current > max) {
            max_pos = i;
            max = current;
        }
    }

    for (int i = 255; i > 128; i--) {
        int current = hist.at<uchar>(i);
        if ((current < max / 12) && onHill) {
            onHill = false;
            output.push_back(i);
        }
        if (current > max / 2) {
            onHill = true;
        }
    }
    if (output.size())
        return output[output.size() - 1];
    else
        return 200;
}


cv::Mat rowMean(cv::Mat& mat) {
    cv::Mat row_mean;
    cv::reduce(mat, row_mean, 0, cv::REDUCE_AVG);
    cv::Mat row_histogram(255, mat.cols, CV_8UC1, cv::Scalar(0));

    for (int i = 1; i < mat.cols; i++) {
        cv::line(
            row_histogram,
            cv::Point(i - 1, 256 - row_mean.at<uchar>(0, i - 1)),
            cv::Point(i, 256 - row_mean.at<uchar>(0, i)),
            cv::Scalar(255, 0, 0), 2, 8, 0);
    }

    return row_histogram;
}

cv::Mat colMean(cv::Mat& mat) {
    cv::Mat col_mean;
    reduce(mat, col_mean, 1, cv::REDUCE_AVG);
    col_mean = col_mean.t();
    cv::Mat col_histogram(255, mat.rows, CV_8UC1, cv::Scalar(0));
    
    for (int i = 1; i < mat.rows; i++) {
        line(
            col_histogram,
            cv::Point(i - 1, 256 - col_mean.at<uchar>(0, i - 1)),
            cv::Point(i, 256 - col_mean.at<uchar>(0, i)),
            cv::Scalar(255, 0, 0), 2, 8, 0);
    }

    return col_histogram;
}

vector<int> get_borders(cv::Mat& mean, uchar white) {
    int width = mean.cols;
    int middle = width / 2;

    vector<int> positions;
    
    for (int i = middle; i > 0; i--) {
        uchar current = mean.at<uchar>(i);
        if (current < white / 3) {
            positions.push_back(i);
            break;
        }
    }

    for (int i = middle; i < width - 1; i++) {
        uchar current = mean.at<uchar>(i);
        if (current < white / 3) {
            positions.push_back(i);
            break;
        }
    }
    if (positions.size() == 2)
        return positions;
    else {
        vector<int> x{ 100, mean.cols - 100 };
        return x;
    }
}

cv::Mat houghTransform(cv::Mat& mat, int thetaAxisSize, int rAxisSize, uint threshold) {
    int width = mat.cols;
    int height = mat.rows;
    int maxRadius = floor(sqrt(pow(width, 2) + pow(height, 2)));
    int halfRAxisSize = floor(rAxisSize / 2);
    uint max = 0;
    
    uint** array = new uint * [thetaAxisSize];
    for (int i = 0; i < thetaAxisSize; i++)
        array[i] = new uint[rAxisSize]{ 0 };
    
    double* sinTable = new double[thetaAxisSize];
    double* cosTable = new double[thetaAxisSize];

    for (int theta = thetaAxisSize - 1; theta >= 0; theta--)
    {
        double thetaRadians = theta * CV_PI / thetaAxisSize;
        sinTable[theta] = sin(thetaRadians);
        cosTable[theta] = cos(thetaRadians);
    }
    
    for (int y = height - 1; y >= 0; y--)
    {
        for (int x = width - 1; x >= 0; x--)
        {
            if (mat.at<uchar>(y,x) == 255)
            {
                for (int theta = thetaAxisSize - 1; theta >= 0; theta--)
                {
                    double r = cosTable[theta] * x + sinTable[theta] * y;
                    int rScaled = (int)floor(r * halfRAxisSize / maxRadius) + halfRAxisSize;
                    array[theta][rScaled] += 1;
                    if (array[theta][rScaled] > max) {
                        max = array[theta][rScaled];
                    }
                }
            }
        }
    }

    cv::Mat output = cv::Mat::zeros(thetaAxisSize, rAxisSize, CV_8UC1);
    
    for (int y = 0; y < rAxisSize; y++) {
        for (int x = 0; x < thetaAxisSize; x++) {
            uint current = array[y][x];
            if (current > threshold) {
                output.at<uchar>(x, rAxisSize - 1 - y) = floor((current * 254.0) / max);
            }
        }
    }
    return output;
}

bool compare3iPointBy3rdValue(cv::Point3i P1, cv::Point3i P2) {
    int i = P1.z;
    int j = P2.z;
    return (i < j);
}

vector<cv::Point3i> getPeaksFromHough(cv::Mat& hough, uchar threshold) {

    vector<cv::Point3i> output;
    for (int i = 0; i < hough.cols; i++) {
        for (int j = 0; j < hough.rows; j++) {
            if (hough.at<uchar>(i,j) > threshold) {
                output.push_back(cv::Point3i(j,i, int(hough.at<uchar>(i,j))));
            }
        }
    }

    sort(output.begin(), output.end(), compare3iPointBy3rdValue);
    return output;
}


int main()
{
    std::string img_path = cv::samples::findFile("C:/Users/Juzek/Desktop/IMG_20220514_123422.jpg");
    cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);

    srand(time(0));

    int width = img.cols;
    int height = img.rows;
    std::cout << "Width:" << width << " Height:" << height << endl;

    cv::Mat biliteral;
    bilateralFilter(img, biliteral, 9, 25, 25);

    cv::Mat biliteral_img;
    cv::Mat biliteral_hist = drawHistogram(biliteral);
    uchar white = getWhite(biliteral_hist);
    cv::Scalar mean_value = mean(biliteral);


    cv::Mat crop, thr;

    
    if (mean_value[0] < white - 40) {
        cv::Mat row_mean;
        cv::Mat col_mean;
        reduce(biliteral, col_mean, 1, cv::REDUCE_AVG);
        reduce(biliteral, row_mean, 0, cv::REDUCE_AVG);
        col_mean = col_mean.t();

        cv::Mat col_mean_img = colMean(biliteral);
        cv::Mat row_mean_img = rowMean(biliteral);
        imshow("col_mean_img", col_mean_img);
        imshow("row_mean_img", row_mean_img);

        vector<int> col_pos = get_borders(col_mean, white);
        vector<int> row_pos = get_borders(row_mean, white);

        vector<cv::Point> paper_points = {
            cv::Point(row_pos[0], col_pos[1]),  // left_top
            cv::Point(row_pos[1], col_pos[1]),  // right_top
            cv::Point(row_pos[1], col_pos[0]),  // right_bottom
            cv::Point(row_pos[0], col_pos[0])   // left_bottom
        };

        line(biliteral, paper_points[0], paper_points[1], cv::Scalar(0), 5);
        line(biliteral, paper_points[1], paper_points[2], cv::Scalar(0), 5);
        line(biliteral, paper_points[2], paper_points[3], cv::Scalar(0), 5);
        line(biliteral, paper_points[3], paper_points[0], cv::Scalar(0), 5);

        resize(biliteral, biliteral_img, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
        imshow("biliteral_img", biliteral_img);
        imshow("biliteral_hist", biliteral_hist);

        cv::Rect crop_rect(paper_points[0], paper_points[2]);
        cv::Mat cropped = biliteral(crop_rect);

        int padding_w = width / 50;
        int padding_h = height / 50;
        crop = cv::Mat(cropped, cv::Rect(padding_w, padding_h, cropped.cols - 2 * padding_w, cropped.rows - 2 * padding_h));
        adaptiveThreshold(crop, thr, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 25, 4);
    }
    else {
        int padding_w = width / 30;
        int padding_h = height / 30;
        biliteral = cv::Mat(biliteral, cv::Rect(padding_w, padding_h, biliteral.cols - 2 * padding_w, biliteral.rows - 2 * padding_h));
        adaptiveThreshold(biliteral, thr, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 21, 4);
    }

    cv::Mat imgResized;
    resize(thr, imgResized, cv::Size(), 0.2, 0.2, cv::INTER_AREA);
    imshow("agssg", imgResized);

    cv::Mat hough = houghTransform(imgResized, 300, 300, 0);
    imshow("hough", hough);

    vector<cv::Point3i> peaks = getPeaksFromHough(hough, 250);

    cout << "White value: " << int(white) << endl;

    for (int i = 0; i < peaks.size(); i++) {
        cout << peaks[i] << endl;
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return 0;
}