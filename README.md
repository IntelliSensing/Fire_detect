# Fire point detection
## 0. Table of Contents
* [Introduction](#1-Introduction)
* [Highlights](#2-Highlights)
* [Model](#3-Model)
* [Dataset](#4-Dataset)
	* [Download](#41-Download)
	* [Requirement](#42-Requirement)
* [Infrared Video Data Processing](#5-Infrared-Video-Data-Processing)
* [Result](#6-Result)
   * [Quantitative Results](#61-Quantitative-Results)
       * [Visible light model evaluation metric](#611-Visible-light-model-evaluation-metric)
       * [Different Infrared model evaluation metric](#612-Different-Infrared-model-evaluation-metric) 
   * [Qualitative Results](#62-Qualitative-Results)
       * [Visible light model results](#621-Visible-light-model-results)
       * [Infrared model results](#622-Infrared-model-results)
## 1. Introduction
&emsp;This code repository focuses on detecting fires from both visible light and infrared images.
## 2. Highlights
* A multi-modal model, including infrared and visible light.<br />
* The lightweight model is trained by using [YOLOv8n](https://github.com/ultralytics/ultralytics) through the D-Fire data set and our own processed flame2 data set, with excellent performance and small parameter amount, which is easy to deploy.<br />
* We used our own frame matching method to optimize the identification of inter-frame leaks in infrared fire point detection. Links to relevant papers will be provided in the future.<br />
## 3. Model
&emsp;We used the YOLOv8 model for training on the self-made infrared dataset and the D-Fire dataset, which can realize multi-modal fire point recognition and smoke detection. While ensuring high accuracy, we adjusted the relevant hyperparameters to ensure that the model also has high efficiency and inference speed.<br>
&emsp;The details of the training model and the test model are given in the [README.md](./train_models) file in the train_models.
## 4. Dataset
### &emsp;4.1 Download
&emsp;The dataset of the visible light model of this project is the processed and data-enhanced [D-Fire dataset](https://github.com/gaiasd/DFireDataset), and we provide the relevant data-augmented code part, the unmodified D-Fire dataset is here. You can also download the processed dataset directly.<br>
&emsp;Visible light dataset：link：[https://pan.baidu.com/s/14C1ePeKg6NYoMlIfsJ9lEg](https://pan.baidu.com/s/14C1ePeKg6NYoMlIfsJ9lEg) 
password：0n87 <br>
&emsp;The infrared dataset is obtained by extracting the infrared video from the [FLAME2 dataset](https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset) through frame extraction and binarization labeling, and we provide the relevant code and also provide the processed dataset.<br>
&emsp;Infrared dataset：link：[https://pan.baidu.com/s/1kl5r-iN5jHN2gYKWQHTdew](https://pan.baidu.com/s/1kl5r-iN5jHN2gYKWQHTdew) 
password：jx3r 
### &emsp;4.2 Requirement
&emsp;ultralytics==8.0.136<br>
&emsp;streamlit==1.24.0<br>
&emsp;py-cpuinfo<br>
&emsp;opencv-python==4.8.1.78<br>
&emsp;numpy==1.24.3<br>
&emsp;matplotlib==3.7.4<br>
&emsp;albumentations==1.3.1<br>
&emsp;torchvision==0.16.0<br>

## 5. Infrared Video Data Processing
<div align="center">
   <img src="https://img2.imgtp.com/2024/05/15/ZzJMvwoS.jpg"  width=912 height=295>
</div>
- The video data is cut into pictures every 10 frames in advance, and then processed.<br/>
- Firstly, the original image is converted into a grayscale image, and then binarization is used to display the fire point in the image.<br/>
- The position of the fire point in the image is obtained through edge detection, and then labeled data is obtained by adjacent box fusion as Dataset v1.<br/>
- Model v1 is trained using Dataset v1, and the knowledge transfer ability of Model v1 is used to recognize the original image.<br/>
- The recognition result is used as Dataset v2 to train Model v2, introducing a Frame Matching Algorithm.<br/>
- Dataset v3 is obtained after processing Dataset v2 with the Frame Matching Algorithm, and finally the final Model v3 is obtained by training Dataset v3.<br/>

<div align="center">
   <img src="https://img2.imgtp.com/2024/05/15/IsxPlaSL.jpg"  width=1280 height=205>
</div>
- Read annotation data of two frames simultaneously: the previous frame and the current picture.<br/>
- Use the area target indicated by the position of the annotation box in the previous frame as the matching target in the current picture.<br/>
- Traverse and match the target frame in the previous frame image to obtain all targets in the previous frame picture.<br/>
- Filter the position in the current frame using the IoU algorithm to remove duplicate marked boxes and retain missing boxes.<br/>
- Sequentially perform this process to utilize spatial information from previous and subsequent frame images.<br/>
- By reversing the input sequence of video images and repeating the process, an image dataset with similar target information in the previous and subsequent frames can be obtained.<br/>


## 6. Result
### &emsp;6.1 Quantitative Results
#### &emsp;&emsp;6.1.1 Visible light model evaluation metric
<div align="center">
<table>
  <tr>
    <th>visible light Detection</th>
    <th>mAP50</th>
    <th>mAP50-95</th>
    <th>precision</th>
    <th>recall</th>
    <th>miss rate</th>
  </tr>
  <tr>
     <td>Fire</td>
    <td>74.7%</td>
    <td>42.8%</td>
    <td>77.0%</td>
     <td>68.1%</td>
     <td>31.9%</td>
  </tr>
  <tr>
    <td>Smoke</td>
    <td>86.1%</td>
    <td>56.1%</td>
    <td>87.8%</td>
     <td>78.0%</td>
     <td>22.0%</td>
  </tr>
</table>
</div>

#### &emsp;&emsp;6.1.2 Different Infrared model evaluation metric
<div align="center">
<table>
  <tr>
    <th>Different IR Fire Detection</th>
    <th>mAP50</th>
    <th>mAP50-95</th>
    <th>precision</th>
    <th>recall</th>
    <th>miss rate</th>
  </tr>
   <tr>
     <td>Model v1</td>
    <td>92.3%</td>
    <td>67%</td>
    <td>86.1%</td>
     <td>87.7%</td>
     <td>12.3%</td>
  </tr>
   <tr>
     <td>Model v2</td>
    <td>91.7%</td>
    <td>64.8%</td>
    <td>86.0%</td>
     <td>87.4%</td>
     <td>12.6%</td>
  </tr>
  <tr>
     <td>Model v3</td>
    <td>93.6%</td>
    <td>71.8%</td>
    <td>88.9%</td>
     <td>86.6%</td>
     <td>13.4%</td>
  </tr>
</table>
</div>
&emsp;In our experiments, Model v1 is trained using a coarsely machine annotated dataset, Model v2 is trained using a dataset annotated by a knowledge transfer model, and Model v3 is trained using a fine-annotated dataset by an image frame matching algorithm. <br/>

### &emsp;6.2 Qualitative Results

#### &emsp;&emsp;6.2.1 Visible light model results
<div align="center">
   <img src="https://img2.imgtp.com/2024/03/01/bfWtK7Z4.jpeg"  width=400 height=250><img src="https://img2.imgtp.com/2024/03/01/Qv3nULPH.jpeg" width=400 height=250>
</div>
<div align="center">
   <img src="https://img2.imgtp.com/2024/03/01/eG54KlXV.jpeg"  width=400 height=250><img src="https://img2.imgtp.com/2024/03/01/YKljm6dF.jpeg" width=400 height=250>
</div>

#### &emsp;&emsp;6.2.2 Comparison of infrared model results
<div align="center">
   <img src="https://img2.imgtp.com/2024/05/15/Cstrdc71.jpg"  width=800 height=500>
</div>
&emsp;(a)(d) is recognized by Model v1, (b)(e) is recognized by Model v2, (c)(f) is recognized by Model v3.The pink box is the fire point recognized by the model, and the green circle is the difference between the images.<br/>

 




