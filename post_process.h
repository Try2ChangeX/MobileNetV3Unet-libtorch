//
// Created by calyx on 2023/2/16.
//

#ifndef UNET_LIBTORCH_POST_PROCESS_H
#define UNET_LIBTORCH_POST_PROCESS_H

#include "opencv2/opencv.hpp"

using namespace cv;

struct detect_box
{
    int id;
    Rect box;
};

struct detect_info
{
    int nums;
    detect_box r[10];
};

detect_info get_pred_box(Mat mask)
{
    detect_info m_detect_info;

    Mat img_labels, img_stats, img_centroids;
    int num = connectedComponentsWithStats(mask, img_labels, img_stats, img_centroids, 4);

    int x1, y1, w, h, area;

    int target_id;
    int max_area = 0;

    m_detect_info.nums = num - 1;

    if (num > 1) {
        for (int k = 1; k < num; k++)
        {
            m_detect_info.r[k - 1].id = k -1;
            m_detect_info.r[k - 1].box.x = img_stats.at<int>(k, 0);
            m_detect_info.r[k - 1].box.y = img_stats.at<int>(k, 1);
            m_detect_info.r[k - 1].box.width = img_stats.at<int>(k, 2);
            m_detect_info.r[k - 1].box.height = img_stats.at<int>(k, 3);
        }
    }


    return m_detect_info;
}




#endif //UNET_LIBTORCH_POST_PROCESS_H
