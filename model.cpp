#include "model.hpp"

using namespace std;
using namespace cv;

Unet::Unet()
{   

}

Unet::~Unet()
{
    
}

void Unet::init(string model_path)  
{
    this->module = torch::jit::load(model_path);
}

Mat Unet::predict(string filedir)
{
    cv::Mat image = cv::imread(filedir);
    std::cout << image.rows <<" " << image.cols <<" " << image.channels() << std::endl;
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
	cv::Mat img_resize;
    resize(image,img_resize, cv::Size(cfg.img_size, cfg.img_size));

    std::cout << img_resize.rows <<" " << img_resize.cols <<" " << img_resize.channels() << std::endl;

    torch::Tensor tensor_img = torch::from_blob(img_resize.data, {1,cfg.img_size,cfg.img_size,3}, torch::kByte);

    tensor_img = tensor_img.permute({0,3,1,2}).toType(torch::kFloat);

    // cout<<"tensor_img:\n"<<tensor_img<<endl;

    torch::Tensor result_tensor = module.forward({tensor_img}).toTensor();
    auto out_tensor = result_tensor.squeeze();

    out_tensor = out_tensor.mul(255).clamp(0, 255).toType(torch::kU8).to(torch::kCPU);

    // cout<<out_tensor<<endl;

    cout<<"here"<<endl;

    cv::Mat resultImg;
    resultImg.create(cv::Size(cfg.img_size /2,cfg.img_size/2), CV_8UC1);
    memcpy((void*)resultImg.data, out_tensor.data_ptr(), out_tensor.numel() * sizeof(torch::kU8));

    return resultImg;
}