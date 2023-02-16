#include <iostream>
#include <opencv2/opencv.hpp> 
#include <string>
#include <torch/script.h>
#include "model.hpp"

#include "post_process.h"
#include "MOT/MOT_detector.h"

using namespace std;
using namespace cv;

int threshold_value = 128;
int max_BINARY_value = 255;

int CNUM = 20;

bool ADD_WEIGHT = 1;

int main(int argc, const char* argv[])
{
    RNG rng(0xFFFFFFFF);
    Scalar_<int> randColor[CNUM];
    for (int i = 0; i < CNUM; i++)
        rng.fill(randColor[i], RNG::UNIFORM, 0, 256);
    MOT_Detector m_MOT_detector;

    string filedir = "../images/from_raw.png";

    //载入模型地址
    string model_path = "../weight/upsample_Epoch_98_cpu.pt";

    // 初始化
    Unet *u_model;

    // 动态创建的时候主要是加载了模型的 param 和 bin 参数
    u_model = new Unet();
    u_model->init(model_path);

    for (int frame = 1; frame < 999; frame++)
    {
        vector<TrackingBox> m_MOTtracking_results;

        cout << "\nFrame:" << frame << endl;
        char filename_str[80];
        sprintf(filename_str, "/media/calyx/Windy/wdy/3B_data/pngs/WavFile_aa_1121/%d.png", frame);

        cv::Mat image = cv::imread(filename_str);
        int rows = image.rows;
        int cols = image.cols;

        // 预测
        Mat resultImg = u_model->predict(image);
        //修正边缘
        resultImg(cv::Rect(0,0,3,112))=0;

        Mat result_mask;
        Mat mask = Mat::zeros(rows, cols, CV_8UC1);

        resize(resultImg, mask, Size(cols, rows));

        threshold(mask, mask, threshold_value, max_BINARY_value, CV_THRESH_BINARY);
#if 0
        Mat mask_zero1 = Mat::zeros(rows, cols, CV_8UC1);
        Mat mask_zero2 = Mat::zeros(rows, cols, CV_8UC1);

        Mat mask3 = Mat::zeros(rows, cols, CV_8UC3);
        Mat aChannels[3];
        aChannels[0] = mask_zero1;
        aChannels[1] = mask_zero2;
        aChannels[2] = mask;
        merge(aChannels, 3, mask3);

        Mat image_with_mask = Mat::zeros(rows, cols, CV_8UC3);

        addWeighted(image, 0.7, mask3, 0.3, 3, image_with_mask);
#endif

        detect_info result_info = get_pred_box(mask);
        printf("result_info.nums：%d\n", result_info.nums);

        vector<TrackingBox> tmp_vec;


        for(int i = 0; i<result_info.nums; i++)
        {
//            rectangle(image, result_info.r[i].box, Scalar(0, 0, 255));

            TrackingBox tmp;
            tmp.id = i;
            tmp.box = result_info.r[i].box;
            tmp_vec.push_back(tmp);
        }

        m_MOTtracking_results = m_MOT_detector.update(tmp_vec);

        for (auto tb : m_MOTtracking_results) {
            cv::rectangle(image, tb.box, randColor[tb.id % CNUM], 2, 8, 0);
            cv::putText(image, "ID:" + std::to_string(tb.id), cv::Point(int(tb.box.x), int(tb.box.y) - 10), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        randColor[tb.id % CNUM], 2);
        };
        cv::putText(image, "Frame: " + std::to_string(frame + 1), cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 0, 255), 2);

        imshow("image", image);
        waitKey(10);
        // cv::imwrite("./result_raw.jpg",result_mask);
    }

	return 0;
}
