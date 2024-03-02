# Content
### &emsp;Welcome to our project, which focuses on the problem of fire detection and aims to train a model to detect fire and smoke.
# Highlights:
### &emsp;1、A multi-modal model, including infrared and visible light.
### &emsp;2、The lightweight model is trained by using YOLOv8n through the D-Fire data set and our own processed flame2 data set, with excellent performance and small parameter amount, which is easy to deploy.
# model
### &emsp;1、train model
    from ultralytics import YOLO
    
    # Load a model
    model = YOLO('yolov8n.pt')  # load an official model
    PROJECT = 'wildfire_identification_data_enhance2.0'  # project name
    NAME = 'test01'  # run name
    
    model.train(
    	data='data.yaml',  # If you want to train an IR model, use flames2.yaml
    	task='detect',
    	epochs=200,
    	verbose=True,
		batch=64,
    	imgsz=640,
    	patience=20,
    	save=True,
    	device='0,1',
    	workers=8,
    	project=PROJECT,
    	name=NAME,
    	cos_lr=True,
    	lr0=0.0001,
    	lrf=0.00001,
    	warmup_epochs=3,
    	warmup_bias_lr=0.000001,
    	optimizer='Adam',
    	seed=42,
    )
### &emsp;2、test model
#### &emsp;&emsp;(1) Calculate the dataset metrics
    from ultralytics import YOLO
    model_path = "best.pt"
    model = YOLO(model_path)
    metrics = model.val(split="val", iou=0.2, conf=0.31)  
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category

#### &emsp;&emsp;（2）Image recognition
    def model_predict(model, image_path, conf_threshold, iou_threshold):
	    model.predict(
		    image_path,
		    conf=conf_threshold,
		    iou=iou_threshold,
		    save=True,
		    show_labels=True,
		    boxes=True,
		    show_conf=True
    	)
    if __name__ == "__main__":
		model_path = "best.pt"
    	model = YOLO(model_path)
		img_path = "your img path"
		model_predict(model, img_path, 0.2, 0.31)
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

