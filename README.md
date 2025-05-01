# Counting fasteners project

## General Information
- Project sử dụng YOLOv11 để tran mô hình nhận diện vật thể (long đen, bu lông,..)
- Sử dụng ONNX runtime để triển khai nhận diện do có ưu thế về tốc độ tính toán

## More details

### Step 1: Clone the project
Use the following command to clone the project from GitHub to your local machine:
  ```bash
  git clone https://github.com/baokieuv/count-fasteners.git
  ```
### Step 2: Install library
Use the following command to clone the project from GitHub to your local machine:
  ```bash
  pip install -r requirements.txt
  ```
### Step 3: Change model path
Navigate to ```/src/detect``` and change the model path and image path to your path in model.py
  ```py
  model_path = "D:/code/projectTest/computer_vision/project2/best.onnx"
  image_path = "D:/code/projectTest/computer_vision/project2/dataset/images/03-1-1_jpg.rf.c7f4301e375532a4674b5381896c30b6.jpg"
  ```
### Step 4: Run the application
After installing the requirements and change the path, run the application with the following command:
  ```bash
  python model.py
  ```
And see the result:
![image](https://github.com/user-attachments/assets/84dafc5a-cedd-4974-82b2-a8c41aeee4d4)

## Main features




