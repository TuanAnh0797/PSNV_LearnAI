import numpy as np
import matplotlib.pyplot as plt
import time

class PolynomialRegressionAdam:
    """
    Thuật toán Polynomial Regression cài đặt với Adam optimizer
    """
    def __init__(self, degree=2, learning_rate=0.01, n_iterations=2000, 
                 beta1=0.9, beta2=0.999, epsilon=1e-8, lambda_reg=0.01):
        self.degree = degree
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.beta1 = beta1  # Hệ số momentum
        self.beta2 = beta2  # Hệ số RMSprop
        self.epsilon = epsilon  # Để tránh chia cho 0
        self.lambda_reg = lambda_reg  # Hệ số regularization
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _transform_features(self, X):
        """Biến đổi đặc trưng ban đầu thành dạng đa thức bậc degree và chuẩn hóa"""
        n_samples = X.shape[0]
        
        # Chuẩn hóa X để tránh các giá trị quá lớn khi tính x^n
        X_normalized = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + self.epsilon)
        
        X_poly = np.ones((n_samples, self.degree + 1))
        
        for i in range(1, self.degree + 1):
            X_poly[:, i] = X_normalized[:, 0] ** i
            
        return X_poly
    
    def fit(self, X, y):
        """Huấn luyện mô hình với dữ liệu X và nhãn y sử dụng Adam optimizer"""
        # Biến đổi đặc trưng thành dạng đa thức và chuẩn hóa
        X_poly = self._transform_features(X)
        
        # Khởi tạo tham số
        n_samples, n_features = X_poly.shape
        self.weights = np.random.randn(n_features) * 0.01  # Khởi tạo ngẫu nhiên nhỏ
        self.bias = 0
        
        # Khởi tạo các biến cho Adam optimizer
        m_w = np.zeros_like(self.weights)  # Momentum cho weights
        v_w = np.zeros_like(self.weights)  # Velocity cho weights
        m_b = 0  # Momentum cho bias
        v_b = 0  # Velocity cho bias
        
        # Gradient Descent với Adam optimizer
        for i in range(self.n_iterations):
            # Dự đoán
            y_predicted = self._predict(X_poly)
            
            # Tính gradient với L2 regularization
            dw = (1/n_samples) * np.dot(X_poly.T, (y_predicted - y)) + (self.lambda_reg * self.weights)
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Cập nhật biased first moment estimate
            m_w = self.beta1 * m_w + (1 - self.beta1) * dw
            m_b = self.beta1 * m_b + (1 - self.beta1) * db
            
            # Cập nhật biased second raw moment estimate
            v_w = self.beta2 * v_w + (1 - self.beta2) * dw**2
            v_b = self.beta2 * v_b + (1 - self.beta2) * db**2
            
            # Hiệu chỉnh bias (bias correction)
            m_w_corrected = m_w / (1 - self.beta1**(i+1))
            m_b_corrected = m_b / (1 - self.beta1**(i+1))
            v_w_corrected = v_w / (1 - self.beta2**(i+1))
            v_b_corrected = v_b / (1 - self.beta2**(i+1))
            
            # Cập nhật tham số
            self.weights -= self.learning_rate * m_w_corrected / (np.sqrt(v_w_corrected) + self.epsilon)
            self.bias -= self.learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)
            
            # Tính cost function và lưu lại
            cost = self._compute_cost(y, y_predicted)
            self.cost_history.append(cost)
            
            # In tiến trình (tùy chọn)
            if (i+1) % 100 == 0:
                print(f'Iteration: {i+1}, Cost: {cost}')
                
            # Kiểm tra điều kiện dừng sớm
            if i > 0 and abs(cost - self.cost_history[-2]) < 1e-6:
                print(f"Đã hội tụ sau {i+1} vòng lặp")
                break
    
    def _predict(self, X_poly):
        """Dự đoán với tham số hiện tại"""
        return np.dot(X_poly, self.weights) + self.bias
    
    def predict(self, X):
        """API dự đoán cho dữ liệu mới"""
        X_poly = self._transform_features(X)
        return self._predict(X_poly)
    
    def _compute_cost(self, y_true, y_predicted):
        """Tính hàm mất mát MSE với L2 regularization"""
        n_samples = len(y_true)
        mse = (1/n_samples) * np.sum((y_true - y_predicted)**2)
        # Thêm L2 regularization
        l2_reg = (self.lambda_reg / (2 * n_samples)) * np.sum(self.weights**2)
        return mse + l2_reg
    
    def score(self, X, y):
        """Tính R² score"""
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u/v
    
    def get_polynomial_equation(self):
        """Trả về phương trình đa thức"""
        equation = f"{self.bias:.4f}"
        for i in range(1, self.degree + 1):
            if self.weights[i] >= 0:
                equation += f" + {self.weights[i]:.4f}x^{i}"
            else:
                equation += f" - {abs(self.weights[i]):.4f}x^{i}"
        return equation

# Hàm tạo dữ liệu mẫu phi tuyến
def generate_nonlinear_data(n_samples=100, noise=1.0, degree=4):
    """Tạo dữ liệu mẫu phi tuyến với nhiễu"""
    X = np.random.rand(n_samples, 1) * 10 - 5  # Dữ liệu từ -5 đến 5
    
    # Tạo dữ liệu theo đa thức bậc cao
    coeffs = np.random.randn(degree + 1) * 0.5
    y = np.zeros(n_samples)
    
    for i, coeff in enumerate(coeffs):
        y += coeff * X.squeeze()**i
    
    # Thêm nhiễu
    y += np.random.randn(n_samples) * noise
    
    return X, y

# Hàm vẽ kết quả
def plot_polynomial_results(X, y, model, title="Polynomial Regression with Adam"):
    """Vẽ dữ liệu và đường hồi quy đa thức"""
    plt.figure(figsize=(10, 6))
    
    # Sắp xếp X để đường cong mượt hơn
    X_sorted = np.sort(X, axis=0)
    y_pred_sorted = model.predict(X_sorted)
    
    # Vẽ dữ liệu
    plt.scatter(X, y, color='blue', label='Dữ liệu')
    
    # Vẽ đường hồi quy
    plt.plot(X_sorted, y_pred_sorted, color='red', linewidth=3, label='Đường hồi quy đa thức')
    
    # Thông tin mô hình
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    equation = model.get_polynomial_equation()
    plt.annotate(f'y = {equation}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10)
    
    plt.show()

# Hàm vẽ đường cong học
def plot_learning_curve(model):
    """Vẽ đường cong học (cost theo số vòng lặp)"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(model.cost_history)+1), model.cost_history, color='blue')
    plt.xlabel('Số vòng lặp')
    plt.ylabel('Cost (MSE)')
    plt.title('Đường cong học')
    plt.grid(True, alpha=0.3)
    plt.show()

# DEMO: Sử dụng mô hình
if __name__ == "__main__":
    # Tạo dữ liệu đa thức bậc cao
    true_degree = 6
    X, y = generate_nonlinear_data(n_samples=150, noise=2.0, degree=true_degree)
    
    start_time = time.time()

    # Huấn luyện mô hình với bậc 8 (cao hơn dữ liệu gốc)
    model_degree = 8
    model = PolynomialRegressionAdam(
        degree=model_degree, 
        learning_rate=0.01, 
        n_iterations=3000,
        beta1=0.9,
        beta2=0.999,
        lambda_reg=0.01
    )
    model.fit(X, y)
    
    # Đánh giá mô hình
    r2_score = model.score(X, y)
    print(f"Mô hình bậc {model_degree} cho dữ liệu bậc {true_degree}")
    print(f"Hệ số R²: {r2_score:.4f}")
    print(f"Phương trình: y = {model.get_polynomial_equation()}")

    end_time = time.time()  # Lấy thời gian kết thúc
    execution_time = end_time - start_time  # Thời gian thực thi
    print(f"Thời gian thực thi: {execution_time} giây")

    # Vẽ kết quả
    plot_polynomial_results(X, y, model, f"Polynomial Regression với Adam (bậc {model_degree})")
    plot_learning_curve(model)

# Hàm so sánh các mô hình với các bậc khác nhau
def compare_polynomial_models(X, y, degrees=[1, 4, 6, 8], learning_rate=0.01, n_iterations=2000):
    """So sánh các mô hình đa thức với các bậc khác nhau"""
    plt.figure(figsize=(15, 10))
    
    # Tạo lưới con
    rows = len(degrees) // 2 + len(degrees) % 2
    cols = min(2, len(degrees))
    
    for i, degree in enumerate(degrees):
        # Huấn luyện mô hình
        model = PolynomialRegressionAdam(
            degree=degree, 
            learning_rate=learning_rate, 
            n_iterations=n_iterations,
            lambda_reg=0.01
        )
        model.fit(X, y)
        
        # Đánh giá mô hình
        r2_score = model.score(X, y)
        
        # Tạo subplot
        plt.subplot(rows, cols, i+1)
        
        # Sắp xếp X để đường cong mượt hơn
        X_sorted = np.sort(X, axis=0)
        y_pred_sorted = model.predict(X_sorted)
        
        # Vẽ dữ liệu
        plt.scatter(X, y, color='blue', s=10, alpha=0.6, label='Dữ liệu')
        
        # Vẽ đường hồi quy
        plt.plot(X_sorted, y_pred_sorted, color='red', linewidth=2, label='Mô hình')
        
        # Thông tin mô hình
        plt.title(f"Bậc {degree}, R² = {r2_score:.4f}")
        plt.grid(True, alpha=0.3)
        
        if i == 0 or i == 2:
            plt.ylabel('y')
        if i >= rows * cols - cols:
            plt.xlabel('X')
    
    plt.tight_layout()
    plt.show()

# Demo so sánh các mô hình
def run_comparison_demo():
    # Tạo dữ liệu
    true_degree = 5
    X, y = generate_nonlinear_data(n_samples=150, noise=2.0, degree=true_degree)
    
    # So sánh các mô hình với các bậc khác nhau
    compare_polynomial_models(
        X, y, 
        degrees=[1, 3, 5, 8], 
        learning_rate=0.01, 
        n_iterations=2000
    )
    
    print(f"Dữ liệu gốc được tạo với đa thức bậc {true_degree}")