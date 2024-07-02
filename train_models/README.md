### Environment configuration
It is recommended to use Python3.8. You can download the corresponding version from the [Python official website](https://www.python.org/).<br/>
Download the requests.txt we provided and run the following code.<br/>
```bash
pip install -r requirements.txt
```
or
```bash
conda install --file requirements.txt
```
### Download the dataset
In order to train our model, we first need to download the dataset, which can be downloaded from Baidu NetDisk. Follow the link we provided to download the dataset locally and unzip it. If you are using Linux or need to install the dataset on a remote server, you can use the bypy library to download the dataset from  baidu netdisk. First, add our dataset to your Baidu Netdisk，then install the library by running the following command:
```bash
pip install bypy
```
Next, authenticate your Baidu NetDisk account:
```bash
bypy info
```
After executing the command, you will get an authentication link, and open the link in a new TAB in the browser that has logged in to the online storage account for authentication. <br>
Copy the authorization code and paste it into the command line. <br>
After the authorization is successful, we will see the bypy folder in the "My App Data" directory in the online storage.<br>
Upload the downloaded files to this folder and execute them on the linux server:
```bash
bypy list 
```
View the files in this directory.<br>
Then execute the command:
```bash
bypy downdir -v
```
Download all files in the current directory.<br>
To unzip the downloaded archive, use the following command:
```bash
unzip D-Fire.zip -d path/to/where/you/want/to/unzip
```
or<br>
```bash
unzip flames2.zip -d path/to/where/you/want/to/unzip
```
### Preprocess the dataset
Once the archive is unzipped, the dataset structure in YOLO format will appear as follows:

```bash
D-Fire
—| train
—---| images
—---| labels
—| test
—---| images
—---| labels
—| val
—---| images
—---| labels
```
The D-fire training set contains 23671 images, while the test set contains 4265 images and the val set contains 5902 images. <br>

```bash
flames2
—| train
—---| images
—---| labels
—| test
—---| images
—---| labels
—| val
—---| images
—---| labels
```
The flames2 training set contains 3660 images, while the test set contains 761 images and the val set contains 646 images. 


### Create configuration files
Before we start training, we need to create configuration files to provide information about the dataset. Create an empty file named "data.yaml" and include the following content:

```python
path: /D-Fire-001/
train: train/images # relative to path
val: val/images # relative to path
test: test/images # relative to path

nc: 2

names: [0: smoke,1: fire]
```
Then create an empty file named "flames.yaml" and include the following content:
```python
path: /flames2/
train: train/images # relative to path
val: val/images # relative to path
test: test/images # relative to path

nc: 1

names: [0: fire]
```
The path specifies the root directory of the dataset, and the train, val, and test paths indicate the relative paths to the corresponding image directories. The names section maps the class IDs to their respective names.

During training, if the Ultralytics library encounters any issues locating your dataset, it will provide informative error messages to help you troubleshoot the problem. In some cases, you might need to adjust the path parameter in the configuration file to ensure the library can find your dataset successfully.


You might be wondering why we are not explicitly specifying the path to the label files. The reason is that the Ultralytics library automatically replaces the 'images' keyword in the provided paths with 'labels' in the training step. Therefore, it is essential to structure your directory as described earlier to ensure the library can locate the corresponding label files correctly. For more information, please refer to [Ultralytics documentation](https://docs.ultralytics.com/datasets/detect/).

### Start training
To install the necessary packages for training, you can use either pip or conda:
```bash
pip install ultralytics
```
or

```bash
conda install ultralytics
```
Training using Ultralytics is straightforward. We will use a Python script for more flexibility in adjusting hyperparameters. More details can be found [here](https://docs.ultralytics.com/modes/train/). <br>
```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model

PROJECT = 'project_name’'  # project name
NAME = 'experiment_name'  # run name

model.train(
   data = 'data.yaml',
   task = 'detect',
   epochs = 200,
   verbose = True,
   batch = 64,
   imgsz = 640,
   patience = 20,
   save = True,
   device = 0,
   workers = 8,
   project = PROJECT,
   name = NAME,
   cos_lr = True,
   lr0 = 0.0001,
   lrf = 0.00001,
   warmup_epochs = 3,
   warmup_bias_lr = 0.000001,
   optimizer = 'Adam',
   seed = 42,
)
```
The data parameter specifies the path to the configuration file we created earlier.So when you train the IR model just replace data.yaml with flames2.yaml. You can adjust the hyperparameters to suit your specific requirements. The Ultralytics documentation provides further details on available hyperparameters ([link](https://docs.ultralytics.com/modes/train/#arguments)).

One important note is that Ultralytics does not provide a parameter to change the metric used to determine the best model during training. By default, it uses precision as the metric. If the precision does not improve within the defined patience value (set to 20 in our example), the model training will stop.


## Results
The selected hyperparameters for training proved to be highly effective, leading to smooth convergence and remarkable results. The model training process completed in approximately 130 epochs, demonstrating its efficiency.<br>

The training phase yielded two checkpoints: the last one for resuming training and the best one, representing the model with the highest precision. These checkpoints are stored in the "project_name/experiment_name/weights" directory in PyTorch format. Evaluating the best model on the test set can be accomplished using the following Python code:

```python
from ultralytics import YOLO

model = YOLO(‘project_name/experiment_name/weights/best.pt’)

model.val(split='test', batch=48, imgsz=640, verbose=True, conf = 0.1, iou = 0.5)
```
As evident in the code snippet, we can specify the split for evaluation. By default, it refers to the data.yaml file created earlier, which contains the dataset information. However, if needed, you can change the dataset used for evaluation by specifying the "data" parameter. You can explore all the available arguments for the evaluation function [here](https://docs.ultralytics.com/modes/val/#arguments).

