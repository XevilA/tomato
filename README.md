# Complete Pipeline สำหรับระบบตรวจจับและเก็บเกี่ยวมะเขือเทศเชอร์รี่อัตโนมัติ

## 1. การเตรียม Dataset

### 1.1 ดาวน์โหลด Base Datasets

```bash
# สร้างโฟลเดอร์โครงการ
mkdir cherry_tomato_harvester
cd cherry_tomato_harvester

# ดาวน์โหลด tomatOD dataset
git clone https://github.com/up2metric/tomatOD.git

# ดาวน์โหลด LaboroTomato dataset
git clone https://github.com/laboroai/LaboroTomato.git

# สร้างโฟลเดอร์สำหรับ dataset รวม
mkdir -p dataset/images
mkdir -p dataset/labels
```

### 1.2 เก็บข้อมูลเพิ่มเติม

สำหรับโครงงานของคุณ ควรเก็บภาพเพิ่มเติมที่มีลักษณะเฉพาะ:

1. **ภาพมะเขือเทศเชอร์รี่**: 
   - ผลสุก (สีแดง 90-100%)
   - ผลกึ่งสุก (สีแดง 30-89%)
   - ผลดิบ (สีเขียว/ขาว)

2. **ภาพกิ่งและก้าน**:
   - กิ่งหลักที่ต่อกับผล
   - จุดตัดที่เหมาะสม
   - มุมมองต่างๆ

### 1.3 การ Label ข้อมูล

#### ใช้ LabelImg สำหรับ Bounding Box:

```bash
# ติดตั้ง LabelImg
pip install labelImg

# เปิดโปรแกรม
labelImg
```

#### Classes ที่ต้อง Label:
```
0: ripe_cherry (มะเขือเทศสุก)
1: semi_ripe_cherry (มะเขือเทศกึ่งสุก)
2: unripe_cherry (มะเขือเทศดิบ)
3: stem (ก้านผล)
4: cutting_point (จุดตัด)
```

#### ใช้ Roboflow (ทางเลือกที่ง่ายกว่า):
1. สมัครที่ https://roboflow.com/
2. Upload ภาพ
3. Label online พร้อม auto-annotation
4. Export เป็น YOLO format

## 2. การเตรียม Development Environment

### 2.1 ติดตั้ง Dependencies

```bash
# สร้าง virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# หรือ venv\Scripts\activate  # Windows

# ติดตั้ง packages
pip install torch torchvision
pip install ultralytics  # สำหรับ YOLOv8
pip install opencv-python
pip install numpy pandas matplotlib
pip install albumentations  # สำหรับ data augmentation
```

### 2.2 โครงสร้างโฟลเดอร์

```
cherry_tomato_harvester/
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── models/
├── utils/
├── config/
└── results/
```

## 3. Data Preprocessing และ Augmentation

### 3.1 Script สำหรับแบ่ง Dataset

```python
# split_dataset.py
import os
import shutil
import random
from sklearn.model_selection import train_test_split

def split_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.2):
    """
    แบ่ง dataset เป็น train, validation, test
    """
    images = [f for f in os.listdir(f"{source_dir}/images") 
              if f.endswith(('.jpg', '.png'))]
    
    # แบ่งข้อมูล
    train_imgs, temp_imgs = train_test_split(images, test_size=1-train_ratio)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5)
    
    # คัดลอกไฟล์
    for split, img_list in [('train', train_imgs), 
                           ('valid', val_imgs), 
                           ('test', test_imgs)]:
        for img in img_list:
            # คัดลอกรูป
            shutil.copy(f"{source_dir}/images/{img}", 
                       f"{dest_dir}/{split}/images/{img}")
            # คัดลอก label
            label = img.replace('.jpg', '.txt').replace('.png', '.txt')
            if os.path.exists(f"{source_dir}/labels/{label}"):
                shutil.copy(f"{source_dir}/labels/{label}", 
                           f"{dest_dir}/{split}/labels/{label}")
```

### 3.2 Data Augmentation

```python
# augmentation.py
import albumentations as A
import cv2

def get_augmentation_pipeline():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, 
                                   contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.2),
        A.RandomShadow(p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, 
                     saturation=0.2, hue=0.1, p=0.5),
        A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0), p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', 
                                label_fields=['class_labels']))
```

## 4. การเทรน Model

### 4.1 Configuration File

```yaml
# config/dataset.yaml
path: ./dataset
train: train/images
val: valid/images
test: test/images

# Classes
names:
  0: ripe_cherry
  1: semi_ripe_cherry
  2: unripe_cherry
  3: stem
  4: cutting_point

nc: 5  # number of classes
```

### 4.2 Training Script

```python
# train.py
from ultralytics import YOLO
import torch

def train_model():
    # เลือก model
    model = YOLO('yolov8m.pt')  # medium model เหมาะสำหรับเริ่มต้น
    
    # การตั้งค่า training
    results = model.train(
        data='config/dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        patience=20,
        save=True,
        device='0' if torch.cuda.is_available() else 'cpu',
        workers=8,
        project='runs/train',
        name='cherry_tomato_v1',
        exist_ok=True,
        pretrained=True,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.0,
        nbs=64,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        auto_augment='randaugment',
        erasing=0.4,
        crop_fraction=1.0
    )
    
    return model

if __name__ == "__main__":
    model = train_model()
```

### 4.3 การทดสอบ Model

```python
# test.py
from ultralytics import YOLO
import cv2
import numpy as np

def test_model(model_path, image_path):
    # โหลด model
    model = YOLO(model_path)
    
    # ทำนายผล
    results = model(image_path)
    
    # วิเคราะห์ผล
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # ดึงข้อมูล
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf
            cls = box.cls
            
            # แสดงผล
            print(f"Class: {model.names[int(cls)]}, Confidence: {conf:.2f}")
    
    # แสดงภาพ
    annotated_frame = results[0].plot()
    cv2.imshow('Detection', annotated_frame)
    cv2.waitKey(0)
```

## 5. การประมวลผลสำหรับ Robotic Arm

### 5.1 การคำนวณตำแหน่งการตัด

```python
# cutting_point_calculator.py
import numpy as np

def calculate_cutting_point(tomato_bbox, stem_bbox):
    """
    คำนวณจุดที่เหมาะสมสำหรับการตัด
    """
    # หาจุดกึ่งกลางของ tomato
    tomato_center = [(tomato_bbox[0] + tomato_bbox[2]) / 2,
                     (tomato_bbox[1] + tomato_bbox[3]) / 2]
    
    # หาจุดที่ใกล้ที่สุดบน stem
    stem_points = extract_stem_points(stem_bbox)
    
    # คำนวณระยะห่างที่เหมาะสม (5-10mm จากผล)
    cutting_distance = 7  # mm
    
    cutting_point = calculate_optimal_point(tomato_center, 
                                           stem_points, 
                                           cutting_distance)
    
    return cutting_point

def convert_to_robot_coordinates(pixel_coords, depth_value, camera_matrix):
    """
    แปลงพิกัด pixel เป็นพิกัด 3D สำหรับ robot
    """
    # Camera calibration parameters
    fx, fy = camera_matrix[0][0], camera_matrix[1][1]
    cx, cy = camera_matrix[0][2], camera_matrix[1][2]
    
    # แปลงเป็น 3D coordinates
    x = (pixel_coords[0] - cx) * depth_value / fx
    y = (pixel_coords[1] - cy) * depth_value / fy
    z = depth_value
    
    return [x, y, z]
```

### 5.2 Integration Script

```python
# main_harvester.py
import cv2
import numpy as np
from ultralytics import YOLO
import serial  # สำหรับสื่อสารกับ Arduino/Robot

class CherryTomatoHarvester:
    def __init__(self, model_path, camera_id=0):
        self.model = YOLO(model_path)
        self.camera = cv2.VideoCapture(camera_id)
        self.robot_serial = serial.Serial('/dev/ttyUSB0', 9600)
        
    def detect_and_harvest(self):
        while True:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            # ตรวจจับ
            results = self.model(frame)
            
            # วิเคราะห์ผล
            ripe_tomatoes = []
            stems = []
            
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls)
                    if cls == 0:  # ripe cherry
                        ripe_tomatoes.append(box.xyxy[0])
                    elif cls == 3:  # stem
                        stems.append(box.xyxy[0])
            
            # ประมวลผลแต่ละผลที่สุก
            for tomato in ripe_tomatoes:
                # หา stem ที่ใกล้ที่สุด
                nearest_stem = self.find_nearest_stem(tomato, stems)
                
                if nearest_stem is not None:
                    # คำนวณจุดตัด
                    cutting_point = calculate_cutting_point(tomato, nearest_stem)
                    
                    # แปลงเป็นพิกัด robot
                    robot_coords = convert_to_robot_coordinates(cutting_point, 
                                                               depth_value, 
                                                               camera_matrix)
                    
                    # ส่งคำสั่งไปยัง robot
                    self.send_to_robot(robot_coords)
            
            # แสดงผล
            annotated = results[0].plot()
            cv2.imshow('Harvester View', annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def find_nearest_stem(self, tomato_bbox, stems):
        """หา stem ที่ใกล้กับ tomato มากที่สุด"""
        min_distance = float('inf')
        nearest = None
        
        tomato_center = [(tomato_bbox[0] + tomato_bbox[2]) / 2,
                        (tomato_bbox[1] + tomato_bbox[3]) / 2]
        
        for stem in stems:
            stem_center = [(stem[0] + stem[2]) / 2,
                          (stem[1] + stem[3]) / 2]
            
            distance = np.sqrt((tomato_center[0] - stem_center[0])**2 + 
                             (tomato_center[1] - stem_center[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest = stem
        
        return nearest
    
    def send_to_robot(self, coordinates):
        """ส่งพิกัดไปยัง robot arm"""
        command = f"MOVE:{coordinates[0]},{coordinates[1]},{coordinates[2]}\n"
        self.robot_serial.write(command.encode())
```

## 6. การ Deploy และทดสอบ

### 6.1 Real-time Testing

```python
# realtime_test.py
def test_realtime():
    harvester = CherryTomatoHarvester('runs/train/cherry_tomato_v1/weights/best.pt')
    harvester.detect_and_harvest()
```

### 6.2 Performance Metrics

```python
# evaluate.py
from ultralytics import YOLO

def evaluate_model(model_path):
    model = YOLO(model_path)
    
    # Validation metrics
    metrics = model.val()
    
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"Precision: {metrics.box.p}")
    print(f"Recall: {metrics.box.r}")
```

## 7. Hardware Integration

### 7.1 Camera Setup
- ใช้ RGB-D camera (เช่น Intel RealSense D435)
- Mount บนหุ่นยนต์หรือรถเข็น

### 7.2 Robot Arm Requirements
- Degrees of freedom: อย่างน้อย 4-5 DOF
- Reach: 40-60 cm
- Payload: 0.5-1 kg
- End effector: Soft gripper หรือ cutting tool

### 7.3 Arduino Code Example

```cpp
// arduino_controller.ino
#include <Servo.h>

Servo baseServo;
Servo shoulderServo;
Servo elbowServo;
Servo wristServo;
Servo gripperServo;

void setup() {
  Serial.begin(9600);
  
  baseServo.attach(9);
  shoulderServo.attach(10);
  elbowServo.attach(11);
  wristServo.attach(12);
  gripperServo.attach(13);
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    
    if (command.startsWith("MOVE:")) {
      // Parse coordinates
      parseAndMove(command);
    } else if (command == "CUT") {
      performCut();
    } else if (command == "GRIP") {
      performGrip();
    }
  }
}

void parseAndMove(String cmd) {
  // แปลง string coordinates เป็น servo angles
  // และควบคุม servo
}

void performCut() {
  // ควบคุมอุปกรณ์ตัด
}

void performGrip() {
  // ควบคุม gripper
}
```

## 8. Tips และ Best Practices

1. **Data Collection**:
   - เก็บภาพในสภาพแสงต่างๆ
   - มุมมองหลากหลาย
   - ระยะห่างต่างๆ

2. **Model Training**:
   - เริ่มด้วย pretrained model
   - Fine-tune hyperparameters
   - Monitor overfitting

3. **Deployment**:
   - ทดสอบใน environment จริง
   - Optimize inference speed
   - Handle edge cases

4. **Safety**:
   - Emergency stop button
   - Collision detection
   - Safe operating zones

## 9. Resources และ References

- YOLOv8 Documentation: https://docs.ultralytics.com/
- OpenCV Python: https://opencv.org/
- RealSense SDK: https://github.com/IntelRealSense/librealsense
- ROS for Robotics: http://www.ros.org/

## 10. Next Steps

1. เริ่มเก็บ dataset ของคุณเอง
2. Label ข้อมูลอย่างละเอียด
3. Train model และ iterate
4. พัฒนา hardware integration
5. ทดสอบในสภาพแวดล้อมจริง
