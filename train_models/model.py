from ultralytics import YOLO

# Load a model
model = YOLO('/home/yy/data/wildfire/general-models/yolov8n.pt')  # load an official model
PROJECT = 'wildfire_identification_data_enhance2.0'  # project name
NAME = 'test01'  # run name

model.train(
    data='/home/yy/data/wildfire/data.yaml',
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
# torch.save(model.state_dict(), 'path/to/model.pt')
