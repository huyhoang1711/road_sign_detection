# Nhận diện Biển báo Giao thông 

Mục tiêu của dự án là xây dựng một hệ thống có khả năng phát hiện và nhận dạng các biển báo giao thông trong ảnh.

## Phần I: Giới thiệu

Nhận diện biển báo giao thông (Traffic Sign Detection) là một bài toán quan trọng trong lĩnh vực thị giác máy tính, với nhiều ứng dụng thực tiễn trong các hệ thống lái xe tự hành (Self-driving Cars) và hệ thống hỗ trợ lái xe nâng cao (Advanced Driver Assistance Systems).

Chương trình được chia thành hai giai đoạn chính:
1.  **Phát hiện (Detection):** Xác định vị trí của biển báo trong ảnh.
2.  **Nhận dạng (Recognition):** Phân loại biển báo đã được phát hiện (ví dụ: 'stop', 'crosswalk', 'speedlimit').

Dự án này sử dụng mô hình Support Vector Machine (SVM) để phân loại và kỹ thuật Sliding Window kết hợp với Pyramid Image để phát hiện đối tượng.

*   **Đầu vào (Input):** Một bức ảnh chứa biển báo giao thông.
*   **Đầu ra (Output):** Tọa độ vị trí và tên (class) của các biển báo có trong ảnh.

## Phần II: Cài đặt chương trình

### 1. Xây dựng mô hình phân loại biển báo giao thông

#### a. Tải bộ dữ liệu
Bộ dữ liệu được sử dụng bao gồm 877 hình ảnh với 4 loại biển báo: 'trafficlight', 'stop', 'speedlimit', và 'crosswalk'. Tuy nhiên, trong quá trình xử lý, lớp 'trafficlight' sẽ được loại bỏ để tập trung vào các biển báo.

Dữ liệu được cấu trúc thành hai thư mục:
*   `images`: Chứa các file ảnh.
*   `annotations`: Chứa các file `.xml` tương ứng, lưu trữ thông tin về tọa độ và lớp của các đối tượng trong ảnh.

#### b. Các thư viện cần thiết
Dự án sử dụng các thư viện Python phổ biến sau:
```python
import time
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from skimage.transform import resize
from skimage import feature
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

#### c. Tiền xử lý và chuẩn bị dữ liệu
1.  **Đọc dữ liệu:** Duyệt qua các file `.xml` để trích xuất tọa độ (`<bndbox>`) và tên (`<name>`) của từng biển báo.
2.  **Trích xuất đặc trưng HOG:** Để mô hình SVM có thể học hiệu quả, mỗi ảnh của biển báo sẽ được chuyển đổi thành một vector đặc trưng bằng thuật toán **Histogram of Oriented Gradients (HOG)**. Quá trình này bao gồm việc chuyển ảnh sang thang độ xám và thay đổi kích thước ảnh về một kích thước cố định (ví dụ: 32x32 pixels) trước khi tính toán HOG.
3.  **Mã hóa nhãn (Encode label):** Các nhãn dạng chuỗi ('stop', 'crosswalk', 'speedlimit') được chuyển đổi thành dạng số (0, 1, 2) bằng `LabelEncoder` của scikit-learn.
4.  **Chia dữ liệu:** Dữ liệu được chia thành hai tập: huấn luyện (train) và xác thực (validation) với tỷ lệ 7:3.
5.  **Chuẩn hóa dữ liệu:** Các vector đặc trưng HOG được chuẩn hóa bằng `StandardScaler` để cải thiện hiệu suất của mô hình SVM.

#### d. Huấn luyện và đánh giá mô hình
*   **Huấn luyện:** Một mô hình **Support Vector Machine (SVM)** với kernel là `rbf` được huấn luyện trên tập dữ liệu đã chuẩn hóa. Tham số `probability=True` được kích hoạt để có thể ước tính xác suất cho mỗi dự đoán.
*   **Đánh giá:** Mô hình sau khi huấn luyện sẽ được đánh giá độ chính xác trên tập xác thực.

### 2. Xây dựng hàm phát hiện đối tượng

#### a. Kỹ thuật Sliding Window
Đây là một kỹ thuật cơ bản để xác định vị trí đối tượng. Một cửa sổ với kích thước xác định sẽ trượt qua tất cả các vị trí trên ảnh. Tại mỗi vị trí, phần ảnh bên trong cửa sổ sẽ được đưa vào mô hình phân loại đã huấn luyện để kiểm tra xem nó có chứa biển báo hay không.

#### b. Kỹ thuật Image Pyramid
Để phát hiện được các biển báo ở nhiều kích thước khác nhau, kỹ thuật Image Pyramid được áp dụng. Kỹ thuật này tạo ra một chuỗi các phiên bản của ảnh gốc với kích thước giảm dần. Sau đó, thuật toán cửa sổ trượt sẽ được áp dụng trên mỗi ảnh trong kim tự tháp, giúp phát hiện cả những vật thể nhỏ.

#### c. Hậu xử lý với Non-Maximum Suppression (NMS)
Quá trình phát hiện có thể tạo ra nhiều khung giới hạn (bounding box) chồng chéo cho cùng một đối tượng. **Non-Maximum Suppression (NMS)** là một kỹ thuật hậu xử lý được sử dụng để giải quyết vấn đề này. NMS sẽ loại bỏ các khung giới hạn có điểm tin cậy (confidence score) thấp hơn và có độ chồng chéo (đo bằng **Intersection over Union - IoU**) cao với khung có điểm tin cậy cao nhất.
