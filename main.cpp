#include <iostream>
#include <opencv2/opencv.hpp> 
#include <string>
#include <torch/script.h>
#include "model.hpp"

using namespace std;

int threshold_value = 128;
int max_BINARY_value = 255;

bool ADD_WEIGHT = 1;

int main(int argc, const char* argv[]) {
  // if (argc != 2) {
  //   printf("Usage: %s  <bmp> \n", argv[0]);
  //   return -1;
  // }

  // string filedir = argv[1];

  string filedir = "./images/from_raw.png";

  //载入模型地址
  string model_path = "./weight/upsample_Epoch_98_cpu.pt";
  // string filedir = "./images/Misc_53.png";

  // 初始化
  Unet *u_model; 
  
  // 动态创建的时候主要是加载了模型的 param 和 bin 参数 
  u_model = new Unet(); 
  u_model->init(model_path);

  // 预测
  Mat resultImg = u_model->predict(filedir);
  // resultImg[resultImg > 128] = 255;
  Mat result_mask;

  threshold(resultImg, result_mask, threshold_value, max_BINARY_value,CV_THRESH_BINARY); 

  // cv::imwrite("./result_raw.jpg",result_mask);
  printf("save done!");

	return 0;
	}