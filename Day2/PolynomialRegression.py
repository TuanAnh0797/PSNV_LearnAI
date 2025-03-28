import numpy as np
import matplotlib.pyplot as plt
import time

class PolynomialRegression:
    """
    Thuật toán Polynomial Regression cài đặt từ đầu
    """
    def __init__(self, degree=2, learning_rate=0.01, n_iterations=2000, lambda_reg=0.01):
    #def __init__(self, degree=2, learning_rate=0.01, n_iterations=3000):
        self.lambda_reg = lambda_reg
        self.degree = degree
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    # def _transform_features(self, X):
    #     """Biến đổi đặc trưng ban đầu thành dạng đa thức bậc degree"""
    #     n_samples = X.shape[0]
    #     X_poly = np.ones((n_samples, self.degree + 1))
        
    #     for i in range(1, self.degree + 1):
    #         X_poly[:, i] = X[:, 0] ** i
            
    #     return X_poly
    # Thêm vào lớp PolynomialRegression
    def _transform_features(self, X):
        n_samples = X.shape[0]
    # Chuẩn hóa X trước khi tính toán
        X_normalized = (X - X.mean()) / X.std()
        X_poly = np.ones((n_samples, self.degree + 1))
    
        for i in range(1, self.degree + 1):
            X_poly[:, i] = X_normalized[:, 0] ** i
        
        return X_poly
    
    
    def fit(self, X, y):
        """Huấn luyện mô hình với dữ liệu X và nhãn y"""
        # Biến đổi đặc trưng thành dạng đa thức
        X_poly = self._transform_features(X)
        
        # Khởi tạo tham số
        n_samples, n_features = X_poly.shape
        self.weights = np.zeros(n_features)
        #self.weights = np.random.randn(n_features) * 0.05
        self.bias = 0
        
        # Gradient Descent
        for i in range(self.n_iterations):
            # Dự đoán
            y_predicted = self._predict(X_poly)
            
            # Tính đạo hàm
            #dw = (1/n_samples) * np.dot(X_poly.T, (y_predicted - y))
            dw = (1/n_samples) * np.dot(X_poly.T, (y_predicted - y)) + (self.lambda_reg * self.weights)
            db = (1/n_samples) * np.sum(y_predicted - y)
            
            # Cập nhật tham số
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Tính cost function và lưu lại
            cost = self._compute_cost(y, y_predicted)
            self.cost_history.append(cost)
            
            # In tiến trình (tùy chọn)
            if (i+1) % 100 == 0:
                print(f'Iteration: {i+1}, Cost: {cost}')
            # Trong vòng lặp Gradient Descent
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
        """Tính hàm mất mát MSE"""
        n_samples = len(y_true)
        cost = (1/n_samples) * np.sum((y_true - y_predicted)**2)
        return cost
    
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
def generate_nonlinear_data(n_samples=100, noise=1.0):
    """Tạo dữ liệu mẫu phi tuyến với nhiễu"""
    X = np.random.rand(n_samples, 1) * 10 - 5  # Dữ liệu từ -5 đến 5
    y = 0.5 * X.squeeze()**3 - 2 * X.squeeze()**2 + 3 * X.squeeze() + 2 + np.random.randn(n_samples) * noise
    return X, y

# Hàm vẽ kết quả
def plot_polynomial_results(X, y, model, title="Polynomial Regression"):
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
    plt.annotate(f'y = {equation}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12)
    
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
    # Tạo dữ liệu
    X, y = generate_nonlinear_data(n_samples=100, noise=5.0)
    
    start_time = time.time()

    # Huấn luyện mô hình
    model = PolynomialRegression(degree=10, learning_rate=0.0001, n_iterations=2000)
    model.fit(X, y)
    
    # Đánh giá mô hình
    r2_score = model.score(X, y)
    print(f"Hệ số R²: {r2_score:.4f}")
    print(f"Phương trình: y = {model.get_polynomial_equation()}")

    end_time = time.time()  # Lấy thời gian kết thúc
    execution_time = end_time - start_time  # Thời gian thực thi
    print(f"Thời gian thực thi: {execution_time} giây")

    # Vẽ kết quả
    plot_polynomial_results(X, y, model, f"Polynomial Regression (bậc {model.degree})")
    plot_learning_curve(model)

# Hàm so sánh các mô hình với các bậc khác nhau
def compare_polynomial_models(X, y, degrees=[1, 2, 3, 4], learning_rate=0.001, n_iterations=3000):
    """So sánh các mô hình đa thức với các bậc khác nhau"""
    plt.figure(figsize=(12, 10))
    
    # Tạo lưới con
    rows = len(degrees) // 2 + len(degrees) % 2
    cols = min(2, len(degrees))
    
    for i, degree in enumerate(degrees):
        # Huấn luyện mô hình
        model = PolynomialRegression(degree=degree, learning_rate=learning_rate, n_iterations=n_iterations)
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