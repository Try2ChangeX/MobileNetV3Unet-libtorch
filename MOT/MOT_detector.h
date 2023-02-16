//
// Created by calyx on 2023/2/15.
//

#ifndef MOVING_TARGET_MOT_DETECTOR_H
#define MOVING_TARGET_MOT_DETECTOR_H

#include "thread"
#include "queue"
#include "mutex"

#include "Sort.h"

class MOT_Detector
{
public:
    MOT_Detector();
    ~MOT_Detector();

public:
    void set_params(int maxAge, int minHits, double iou_th);

    void mot_process_task(void* p);

    void run();

    vector<TrackingBox> update(vector<TrackingBox> detFrameData);

    vector<TrackingBox> get_MOT_results();

public:

    int m_max_age;
    int m_min_hits;
    double m_iou_Threshold;

    thread task;
    bool m_stop;

    mutex m_mutex;

    queue<vector<TrackingBox>> m_queue_trackingbox_vector;

    vector<TrackingBox> m_MOTtracking_results;

    Sort* m_MOT_tracker;

};

#endif //MOVING_TARGET_MOT_DETECTOR_H
