#Đã dùng 94% bộ nhớ … Nếu hết dung lượng lưu trữ, bạn sẽ không thể tạo, chỉnh sửa và tải tệp lên. Sử dụng 100 GB dung lượng với giá 45.000 ₫ 11.250 ₫/tháng trong 3 tháng.
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.feature import hog, local_binary_pattern
from skimage.measure import regionprops
from skimage.filters import sobel
from scipy.spatial import distance

class FaceDetector:
    def __init__(self):
        # Các thông số mặc định
        self.skin_lower_hsv = np.array([0, 20, 80], dtype=np.uint8)
        self.skin_upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
        self.skin_lower_hsv2 = np.array([170, 20, 80], dtype=np.uint8)
        self.skin_upper_hsv2 = np.array([180, 255, 255], dtype=np.uint8)
        self.min_face_size = (50, 50)
        self.max_face_size = (300, 300)
        self.aspect_ratio_range = (0.8, 1.4)  # Tỉ lệ khung mặt thường là gần vuông
        self.iou_threshold = 0.5
        
    def read_image(self, image_path):
        """Đọc ảnh và chuyển đổi sang các không gian màu cần thiết"""
        self.orig_img = cv2.imread(image_path)
        if self.orig_img is None:
            raise ValueError(f"Không thể đọc ảnh từ: {image_path}")
        
        self.img = self.orig_img.copy()
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        self.ycrcb = cv2.cvtColor(self.img, cv2.COLOR_BGR2YCrCb)
        return self.img
    
    def extract_features(self):
        """Trích xuất các đặc trưng cạnh, góc và các điểm đặc trưng"""
        # 1. Trích xuất cạnh sử dụng bộ lọc Sobel và Canny
        self.edges = cv2.Canny(self.gray, 50, 150)
        self.sobel_x = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=3)
        self.sobel_y = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=3)
        self.gradient_magnitude = np.sqrt(np.square(self.sobel_x) + np.square(self.sobel_y))
        self.gradient_magnitude = cv2.normalize(self.gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # 2. Trích xuất góc bằng phương pháp Harris
        self.corners = cv2.cornerHarris(self.gray, blockSize=2, ksize=3, k=0.04)
        self.corners = cv2.dilate(self.corners, None)
        _, self.corners_thresh = cv2.threshold(self.corners, 0.01 * self.corners.max(), 255, 0)
        self.corners_thresh = np.uint8(self.corners_thresh)
        
        # 3. Tìm các điểm đặc trưng bằng SIFT hoặc ORB (SIFT đòi hỏi bản quyền, ORB miễn phí)
        self.feature_detector = cv2.ORB_create()
        self.keypoints, self.descriptors = self.feature_detector.detectAndCompute(self.gray, None)
        
        # 4. Trích xuất HOG (Histogram of Oriented Gradients)
        self.hog_features = hog(self.gray, orientations=9, pixels_per_cell=(8, 8), 
                               cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')
        
        # 5. Trích xuất LBP (Local Binary Pattern) - tốt cho nhận dạng khuôn mặt
        radius = 3
        n_points = 8 * radius
        self.lbp = local_binary_pattern(self.gray, n_points, radius, method='uniform')
        
        return {
            'edges': self.edges,
            'corners': self.corners_thresh,
            'keypoints': self.keypoints,
            'hog_features': self.hog_features,
            'lbp': self.lbp
        }
    
    def adaptive_skin_detection(self):
        """Tách vùng màu da bằng nhiều phương pháp kết hợp"""
        # 1. Phát hiện màu da dựa trên ngưỡng HSV (phương pháp cơ bản)
        mask_hsv1 = cv2.inRange(self.hsv, self.skin_lower_hsv, self.skin_upper_hsv)
        mask_hsv2 = cv2.inRange(self.hsv, self.skin_lower_hsv2, self.skin_upper_hsv2)
        mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)
        
        # 2. Phát hiện màu da dựa trên không gian màu YCrCb
        lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
        upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(self.ycrcb, lower_ycrcb, upper_ycrcb)
        
        # 3. Kết hợp 2 phương pháp
        mask_combined = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        
        # 4. Áp dụng adaptive thresholding để cải thiện phân đoạn
        # Lấy kênh Y (độ sáng) từ YCrCb
        y_channel = self.ycrcb[:,:,0]
        mask_adaptive = cv2.adaptiveThreshold(
            y_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # 5. Tinh chỉnh mask bằng các phép toán hình thái học
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_final = cv2.bitwise_and(mask_combined, mask_adaptive)
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel)
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel)
        
        # 6. Adaptive skin detection bằng K-means clustering
        # Reshape ảnh để áp dụng K-means
        reshaped_img = self.img.reshape((-1, 3))
        reshaped_img = np.float32(reshaped_img)
        
        # Áp dụng K-means với k=3 (nền, da, và các vùng khác)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 3
        _, labels, centers = cv2.kmeans(reshaped_img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_img = centers[labels.flatten()]
        segmented_img = segmented_img.reshape(self.img.shape)
        
        # Xác định cluster nào tương ứng với màu da bằng cách so sánh với mask đã biết
        mask_overlap = np.zeros(k, dtype=np.float32)
        for i in range(k):
            cluster_mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
            cluster_mask[labels.reshape(self.img.shape[:2]) == i] = 255
            overlap = cv2.bitwise_and(cluster_mask, mask_combined)
            mask_overlap[i] = cv2.countNonZero(overlap) / max(1, cv2.countNonZero(cluster_mask))
        
        skin_cluster = np.argmax(mask_overlap)
        skin_mask_kmeans = np.zeros(self.img.shape[:2], dtype=np.uint8)
        skin_mask_kmeans[labels.reshape(self.img.shape[:2]) == skin_cluster] = 255
        
        # 7. Kết hợp tất cả các phương pháp
        mask_final = cv2.bitwise_or(mask_final, skin_mask_kmeans)
        
        self.skin_mask = mask_final
        return self.skin_mask
    
    def extract_face_candidates(self):
        """Tách các vùng có thể là khuôn mặt từ vùng màu da"""
        # Tìm contours từ mask màu da
        contours, _ = cv2.findContours(self.skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Lọc các contour dựa trên kích thước và tỉ lệ
        face_candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Kiểm tra kích thước tối thiểu và tối đa
            if (w < self.min_face_size[0] or h < self.min_face_size[1] or 
                w > self.max_face_size[0] or h > self.max_face_size[1]):
                continue
                
            # Kiểm tra tỉ lệ khung hình
            aspect_ratio = w / h
            if (aspect_ratio < self.aspect_ratio_range[0] or 
                aspect_ratio > self.aspect_ratio_range[1]):
                continue
                
            face_candidates.append((x, y, w, h))
            
        self.face_candidates = face_candidates
        return face_candidates
    
    def verify_face_candidates(self):
        """Xác minh các vùng ứng viên có phải là khuôn mặt không dựa trên đặc trưng"""
        verified_faces = []
        
        for (x, y, w, h) in self.face_candidates:
            # Trích xuất ROI từ các đặc trưng đã tính
            roi_gray = self.gray[y:y+h, x:x+w]
            roi_edges = self.edges[y:y+h, x:x+w]
            roi_lbp = self.lbp[y:y+h, x:x+w]
            
            # Kiểm tra mật độ cạnh trong ROI
            edge_density = np.count_nonzero(roi_edges) / (roi_edges.shape[0] * roi_edges.shape[1])
            
            # Đếm số điểm keypoint trong ROI
            kp_in_roi = 0
            for kp in self.keypoints:
                kp_x, kp_y = int(kp.pt[0]), int(kp.pt[1])
                if x <= kp_x <= x+w and y <= kp_y <= y+h:
                    kp_in_roi += 1
            
            kp_density = kp_in_roi / (w * h)
            
            # Tính histogram của LBP trong ROI
            lbp_hist, _ = np.histogram(roi_lbp.ravel(), bins=np.arange(0, 60), range=(0, 59))
            lbp_hist = lbp_hist.astype('float') / (roi_lbp.shape[0] * roi_lbp.shape[1])
            
            # Điểm số mặt dựa trên các đặc trưng
            face_score = 0
            
            # 1. Kiểm tra mật độ cạnh (khuôn mặt có nhiều đường nét)
            if 0.05 < edge_density < 0.3:
                face_score += 1
            
            # 2. Kiểm tra mật độ keypoint (khuôn mặt có nhiều điểm đặc trưng)
            if kp_density > 0.001:
                face_score += 1
            
            # 3. Kiểm tra LBP (so sánh với histogram LBP điển hình của khuôn mặt)
            # Đây chỉ là giả lập, trong thực tế nên dùng classifier học máy
            lbp_face_signature = np.array([0.1, 0.2, 0.3, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1])
            lbp_sample = lbp_hist[:min(9, len(lbp_hist))]
            if len(lbp_sample) == len(lbp_face_signature):
                lbp_distance = distance.euclidean(lbp_sample, lbp_face_signature)
                if lbp_distance < 0.5:  # Ngưỡng tự điều chỉnh
                    face_score += 1
            
            # 4. Kiểm tra tỉ lệ giữa chiều rộng và chiều cao
            if 0.9 < w/h < 1.2:  # Khuôn mặt thường có tỉ lệ gần 1:1
                face_score += 1
                
            # Nếu đạt điểm số tối thiểu, coi là khuôn mặt
            if face_score >= 2:
                verified_faces.append((x, y, w, h))
                
        self.verified_faces = verified_faces
        return verified_faces
    
    def apply_nms(self, boxes, threshold=0.5):
        """Non-Maximum Suppression để loại bỏ các box chồng chéo"""
        if len(boxes) == 0:
            return []
        
        # Sắp xếp boxes theo diện tích (từ lớn đến nhỏ)
        areas = [(w * h) for (_, _, w, h) in boxes]
        sorted_indices = np.argsort(areas)[::-1]
        
        keep = []
        while len(sorted_indices) > 0:
            # Chọn box lớn nhất
            current_idx = sorted_indices[0]
            keep.append(current_idx)
            
            # Tính IoU với các box còn lại
            ious = []
            for idx in sorted_indices[1:]:
                iou = self.calculate_iou(boxes[current_idx], boxes[idx])
                ious.append(iou)
            
            # Giữ lại các box có IoU dưới ngưỡng
            remain_indices = [sorted_indices[i+1] for i, iou in enumerate(ious) if iou <= threshold]
            sorted_indices = remain_indices
        
        return [boxes[i] for i in keep]
    
    def calculate_iou(self, box1, box2):
        """Tính Intersection over Union (IoU) giữa hai box"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Tính tọa độ của vùng giao nhau
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        # Kiểm tra xem có giao nhau không
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Tính diện tích giao nhau
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Tính diện tích các box
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        # Tính IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou
    
    def finalize_detection(self):
        """Áp dụng các kỹ thuật cuối cùng để tối ưu kết quả phát hiện"""
        # 1. Áp dụng NMS để loại bỏ các box chồng chéo
        final_faces = self.apply_nms(self.verified_faces, threshold=self.iou_threshold)
        
        # 2. Mở rộng kích thước các box mặt (mặt có thể lớn hơn vùng da phát hiện)
        expanded_faces = []
        for (x, y, w, h) in final_faces:
            # Mở rộng 10% mỗi chiều
            expand_w = int(w * 0.1)
            expand_h = int(h * 0.1)
            
            new_x = max(0, x - expand_w)
            new_y = max(0, y - expand_h)
            new_w = min(self.img.shape[1] - new_x, w + 2 * expand_w)
            new_h = min(self.img.shape[0] - new_y, h + 2 * expand_h)
            
            expanded_faces.append((new_x, new_y, new_w, new_h))
        
        # 3. Cuối cùng, lọc lại các box dựa trên vị trí tương đối (ví dụ loại bỏ khuôn mặt bên trong khuôn mặt khác)
        self.final_faces = expanded_faces
        return expanded_faces
    
    def draw_results(self):
        """Vẽ kết quả lên ảnh gốc"""
        result_img = self.orig_img.copy()
        
        # Vẽ vùng màu da
        skin_vis = cv2.bitwise_and(self.img, self.img, mask=self.skin_mask)
        
        # Vẽ các khuôn mặt phát hiện được
        for (x, y, w, h) in self.final_faces:
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return result_img, skin_vis
    
    def detect(self, image_path):
        """Thực hiện toàn bộ quá trình phát hiện khuôn mặt"""
        # 1. Đọc ảnh
        self.read_image(image_path)
        
        # 2. Trích xuất đặc trưng
        features = self.extract_features()
        
        # 3. Phát hiện vùng màu da
        skin_mask = self.adaptive_skin_detection()
        
        # 4. Trích xuất các ứng viên khuôn mặt
        face_candidates = self.extract_face_candidates()
        
        # 5. Xác minh các ứng viên
        verified_faces = self.verify_face_candidates()
        
        # 6. Hoàn thiện kết quả
        final_faces = self.finalize_detection()
        
        # 7. Vẽ kết quả
        result_img, skin_vis = self.draw_results()
        
        return result_img, skin_vis, final_faces

# Hàm test
def test_face_detector(image_path):
    detector = FaceDetector()
    result_img, skin_vis, faces = detector.detect(image_path)
    
    # Hiển thị kết quả
    plt.figure(figsize=(15, 10))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(detector.orig_img, cv2.COLOR_BGR2RGB))
    plt.title('Ảnh gốc')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(skin_vis, cv2.COLOR_BGR2RGB))
    plt.title('Vùng màu da')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Kết quả phát hiện: {len(faces)} khuôn mặt')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return detector

# Ví dụ sử dụng
if __name__ == "__main__":
    # Thay đổi đường dẫn ảnh tương ứng
    image_path = r"C:\Users\Administrator\Desktop\StudyAI\Day5\Datasets\Amr_Moussa\Amr_Moussa_0001.jpg"
    detector = test_face_detector(image_path)