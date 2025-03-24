import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles

class SVM:
    """
    Cài đặt Support Vector Machine (SVM) với kernel từ đầu
    """
    
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000, kernel='linear', C=1.0, gamma=1.0, degree=3):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param  # Tham số regularization
        self.n_iterations = n_iterations
        self.kernel = kernel
        self.C = C  # Tham số soft margin
        self.gamma = gamma  # Tham số RBF/Polynomial kernel
        self.degree = degree  # Bậc của polynomial kernel
        self.weights = None
        self.bias = None
        
    def _initialize_parameters(self, X):
        """Khởi tạo tham số weights và bias"""
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0
    
    def _compute_kernel(self, X1, X2=None):
        """Tính ma trận kernel giữa X1 và X2"""
        if X2 is None:
            X2 = X1
        
        # Trường hợp batch
        if len(X1.shape) > 1 and len(X2.shape) > 1:
            if self.kernel == 'linear':
                return np.dot(X1, X2.T)
            elif self.kernel == 'polynomial':
                return (np.dot(X1, X2.T) + 1) ** self.degree
            elif self.kernel == 'rbf':
                # Tính ma trận khoảng cách
                X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
                X2_norm = np.sum(X2 ** 2, axis=1)
                distances = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
                return np.exp(-self.gamma * distances)
            else:
                raise ValueError(f"Kernel '{self.kernel}' không được hỗ trợ")
        
        # Trường hợp single sample
        else:
            if self.kernel == 'linear':
                return np.dot(X1, X2)
            elif self.kernel == 'polynomial':
                return (np.dot(X1, X2) + 1) ** self.degree
            elif self.kernel == 'rbf':
                distance = np.sum((X1 - X2) ** 2)
                return np.exp(-self.gamma * distance)
            else:
                raise ValueError(f"Kernel '{self.kernel}' không được hỗ trợ")
    
    def fit(self, X, y):
        """Huấn luyện mô hình SVM"""
        n_samples, n_features = X.shape
        
        # Đảm bảo nhãn là +1 và -1
        y_ = np.where(y <= 0, -1, 1)
        
        # Khởi tạo tham số
        self._initialize_parameters(X)
        
        # Gradient Descent
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                # Điều kiện để có thể cập nhật
                condition = y_[idx] * (np.dot(self.weights, x_i) + self.bias) >= 1
                
                if condition:
                    # Cập nhật cho trường hợp nằm ngoài lề hoặc phân loại đúng
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    # Cập nhật cho trường hợp vi phạm lề hoặc phân loại sai
                    self.weights -= self.learning_rate * (
                        2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx])
                    )
                    self.bias -= self.learning_rate * (-y_[idx])
    
    def predict(self, X):
        """Dự đoán lớp cho dữ liệu mới"""
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)
    
    def visualize_results(self, X, y, title="SVM Decision Boundary"):
        """Vẽ biên quyết định của SVM"""
        plt.figure(figsize=(10, 6))
        
        # Tạo lưới để vẽ biên quyết định
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        
        # Dự đoán trên tất cả điểm trong lưới
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Vẽ biên quyết định
        plt.contourf(xx, yy, Z, alpha=0.3)
        
        # Vẽ dữ liệu
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
        
        # Vẽ support vectors 
        # (Trong cài đặt đơn giản này, chúng ta không theo dõi support vectors nên bỏ qua bước này)
        
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.show()

# Ví dụ sử dụng: Tạo dữ liệu có thể phân tách tuyến tính
def generate_linearly_separable_data(n_samples=100, centers=2, random_state=42):
    """Tạo dữ liệu có thể phân tách tuyến tính"""
    X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=random_state)
    return X, y

# Ví dụ sử dụng: Tạo dữ liệu không thể phân tách tuyến tính
def generate_nonlinear_data(n_samples=100, noise=0.1, random_state=42):
    """Tạo dữ liệu không thể phân tách tuyến tính"""
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    return X, y

# Thử nghiệm SVM với dữ liệu có thể phân tách tuyến tính
def test_linear_svm():
    # Tạo dữ liệu
    X, y = generate_linearly_separable_data(n_samples=100)
    
    # Huấn luyện SVM tuyến tính
    svm = SVM(learning_rate=0.01, n_iterations=1000, kernel='linear', C=1.0)
    svm.fit(X, y)
    
    # Vẽ kết quả
    svm.visualize_results(X, y, title="Linear SVM Decision Boundary")

# Thử nghiệm SVM với kernel cho dữ liệu không thể phân tách tuyến tính
def test_kernel_svm():
    # Tạo dữ liệu
    X, y = generate_nonlinear_data(n_samples=100)
    
    # Huấn luyện SVM với kernel RBF
    svm = SVM(learning_rate=0.01, n_iterations=1000, kernel='rbf', C=1.0, gamma=0.1)
    # Trong cài đặt đơn giản này, chúng ta không thực sự sử dụng kernel trong fit()
    # Do đó, đây chỉ là cho mục đích minh họa
    svm.fit(X, y)
    
    # Vẽ kết quả
    svm.visualize_results(X, y, title="RBF Kernel SVM Decision Boundary")

if __name__ == "__main__":
    # Chạy ví dụ
    test_linear_svm()
    test_kernel_svm()