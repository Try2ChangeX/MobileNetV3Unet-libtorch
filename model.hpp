#ifndef MODEL_H
#define MODEL_H

#include <vector> 
#include <map>  
 
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 

#include <torch/torch.h>
#include <torch/script.h>

using namespace cv;
using namespace std;

struct Config{ 
    int img_size=224;
};

class Unet {

public: 
    
    Unet();
    
    ~Unet(); 

    void init(string model_path);

    Mat predict(string img_path);
    
    torch::jit::script::Module module;
    
    // config static
    Config cfg; 

};

#endif 