# This is a OpenCV and YOLO object detection tutorial designed for MAE162E

## installation
Clone the repository
```
git clone https://github.com/Kevin75311/UCLA_MAE162E_week2_materials.git
```
Install the environment from the yml file
```
conda env create -f week2.yml
```

## usage 
* OpenCV examples
For each example, please use "cd" to navigate to the directory of their directory.
```
cd ./OpenCV/example1
```
Then we could just run the example.
```
python main.py
```
* YOLO example
Enter the YOLOv4 directory
```
cd ./YOLOv4
```
Install gdown
```
conda install gdown
```
Enter the weights directory
```
cd ./weights
```
Download the weight from google drive
```
gdown 18RjNcN-wpUMie2FpkYxdG0MMrF0rP5Ns
```
Go back to outside
```
cd ..
```
Then we could just run the example.
```
python main.py
```