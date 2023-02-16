//
// Created by calyx on 2023/2/15.
//

#include "MOT_detector.h"

MOT_Detector::MOT_Detector()
{
    m_max_age = 5;
    m_min_hits = 1;
    m_iou_Threshold = 0.1;
    m_MOT_tracker = new Sort(m_max_age, m_min_hits, m_iou_Threshold);

//    run();
}

MOT_Detector::~MOT_Detector()
{
    delete m_MOT_tracker;
    m_stop = true;
}

void MOT_Detector::set_params(int maxAge, int minHits, double iou_th)
{
    m_max_age = maxAge;
    m_min_hits = minHits;
    m_iou_Threshold = iou_th;
    printf("多目标检测超参设置完毕\n");
}

void MOT_Detector::mot_process_task(void* p)
{
    printf("target detect");
    MOT_Detector* obj = (MOT_Detector*)p;

    while (m_stop)
    {
        // 数据从哪里来, 还是从外部的queue传进来
        while(!m_queue_trackingbox_vector.empty())
        {
            // 数据pop出来
            vector<TrackingBox> tmpVec;
            m_mutex.lock();
            tmpVec = m_queue_trackingbox_vector.front();
            m_queue_trackingbox_vector.pop();
            m_mutex.unlock();

            m_MOTtracking_results = m_MOT_tracker->update(tmpVec);
        }
    }
}

void MOT_Detector::run()
{
    m_stop=false;
    task=std::thread(&MOT_Detector::mot_process_task,this,(void*)(this));
}

vector<TrackingBox> MOT_Detector::update(vector<TrackingBox> detFrameData)
{
    m_MOTtracking_results = m_MOT_tracker->update(detFrameData);
    return m_MOTtracking_results;
}

vector<TrackingBox> MOT_Detector::get_MOT_results()
{
    return m_MOTtracking_results;
}