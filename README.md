# Content
### &emsp;Welcome to our project, which focuses on the problem of fire detection and aims to train a model to detect fire and smoke.
# Highlights:
### &emsp;1、A multi-modal model, including infrared and visible light.
### &emsp;2、The lightweight model is trained by using [YOLOv8n](https://github.com/ultralytics/ultralytics) through the D-Fire data set and our own processed flame2 data set, with excellent performance and small parameter amount, which is easy to deploy.
# model
&emsp;We used the YOLOv8 model for training on the self-made infrared dataset and the D-Fire dataset, which can realize multi-modal fire point recognition and smoke detection. While ensuring high accuracy, we adjusted the relevant hyperparameters to ensure that the model also has high efficiency and inference speed.<br>
### &emsp;The details of the training model and the test model are given in the [README.md](./train_models) file in the train_models.
# Test and Val result
### &emsp;&emsp;&emsp;Below is the output data for the visible light model.

<div align="center">
   <img src="https://img2.imgtp.com/2024/03/03/jF05ns0O.jpg">
   <p>0.3ms preprocess, 12.3ms inference, 0.0ms loss, 0.1ms postprocess per image.</p>
</div>

### &emsp;&emsp;&emsp;Below is the output data for the infrared model.

<div align="center">
   <img src="https://img2.imgtp.com/2024/03/03/H4abyJtw.jpg">
   <p>0.4ms preprocess, 16.6ms inference, 0.0ms loss, 0.1ms postprocess per image.</p>
</div>

# Dataset
&emsp;The dataset of the visible light model of this project is the processed and data-enhanced [D-Fire dataset](https://github.com/gaiasd/DFireDataset), and we provide the relevant data-augmented code part, the unmodified D-Fire dataset is here. You can also download the processed dataset directly.<br>
&emsp;The infrared dataset is obtained by extracting the infrared video from the [FLAME2 dataset](https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset) through frame extraction and binarization labeling, and we provide the relevant code and also provide the processed dataset.<br>
&emsp;Visible light dataset：link：[https://pan.baidu.com/s/14C1ePeKg6NYoMlIfsJ9lEg](https://pan.baidu.com/s/14C1ePeKg6NYoMlIfsJ9lEg) 
password：0n87 <br>
&emsp;Infrared dataset：link：[https://pan.baidu.com/s/11qoWdXmQlKBddi3HIAmb-w](https://pan.baidu.com/s/11qoWdXmQlKBddi3HIAmb-w) 
password：qr92 
# requestments
&emsp;ultralytics==8.0.136<br>
&emsp;streamlit==1.24.0<br>
&emsp;py-cpuinfo<br>
&emsp;opencv-python==4.8.1.78<br>
&emsp;numpy==1.24.3<br>
&emsp;matplotlib==3.7.4<br>
&emsp;albumentations==1.3.1<br>
&emsp;torchvision==0.16.0<br>
# result
## &emsp;Visible light model results
<div align="center">
   <img src="https://img2.imgtp.com/2024/03/01/bfWtK7Z4.jpeg"  width=400 height=250><img src="https://img2.imgtp.com/2024/03/01/Qv3nULPH.jpeg" width=400 height=250>
</div>
<div align="center">
   <img src="https://img2.imgtp.com/2024/03/01/eG54KlXV.jpeg"  width=400 height=250><img src="https://img2.imgtp.com/2024/03/01/YKljm6dF.jpeg" width=400 height=250>
</div>

## &emsp;Infrared model results
<div align="center">
   <img src="https://img2.imgtp.com/2024/03/02/DJHwue8S.jpg"  width=400 height=250><img src="https://img2.imgtp.com/2024/03/02/Avix2KkG.jpg" width=400 height=250>
</div>
<div align="center">
   <img src="https://img2.imgtp.com/2024/03/02/uLkp0GdH.jpg"  width=400 height=250><img src="https://img2.imgtp.com/2024/03/02/DP7DpUQ7.jpg" width=400 height=250>
</div>

# Disclaimer<br>
&emsp;Although our model performs well on the D-fire dataset, we do not guarantee that it can perform well in any realistic work environment.
# Others

