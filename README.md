# Nhận diện Biển báo Giao thông 

Mục tiêu của dự án là xây dựng một hệ thống có khả năng phát hiện và nhận dạng các biển báo giao thông trong ảnh.

## Giới thiệu

Nhận diện biển báo giao thông (Traffic Sign Detection) là một bài toán quan trọng trong lĩnh vực thị giác máy tính, với nhiều ứng dụng thực tiễn trong các hệ thống lái xe tự hành (Self-driving Cars) và hệ thống hỗ trợ lái xe nâng cao (Advanced Driver Assistance Systems).

Chương trình được chia thành hai giai đoạn chính:
1.  **Phát hiện (Detection):** Xác định vị trí của biển báo trong ảnh.
2.  **Nhận dạng (Recognition):** Phân loại biển báo đã được phát hiện (ví dụ: 'stop', 'crosswalk', 'speedlimit').

Dự án này sử dụng mô hình Support Vector Machine (SVM) để phân loại và kỹ thuật Sliding Window kết hợp với Pyramid Image để phát hiện đối tượng.
