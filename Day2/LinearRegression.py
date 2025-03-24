import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
class LinearRegression:
    """
    Thuật toán Linear Regression cài đặt từ đầu
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """Huấn luyện mô hình với dữ liệu X và nhãn y"""
        # Khởi tạo tham số
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient Descent
        for i in range(self.n_iterations):
            # Dự đoán
            y_predicted = self._predict(X)
            
            # Tính đạo hàm
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
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
    
    def _predict(self, X):
        """Dự đoán với tham số hiện tại"""
        return np.dot(X, self.weights) + self.bias
    
    def predict(self, X):
        """API dự đoán cho dữ liệu mới"""
        return self._predict(X)
    
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

# Hàm tạo dữ liệu mẫu
def generate_data(n_samples=100, noise=10):
    """Tạo dữ liệu mẫu với nhiễu"""
    X = np.random.rand(n_samples, 1) * 10
    y = 2 * X.squeeze() + 5 + np.random.randn(n_samples) * noise
    return X, y

# Hàm vẽ kết quả
def plot_results(X, y, model, title="Linear Regression"):
    """Vẽ dữ liệu và đường hồi quy"""
    plt.figure(figsize=(10, 6))
    
    # Vẽ dữ liệu
    plt.scatter(X, y, color='blue', label='Dữ liệu')
    
    # Vẽ đường hồi quy
    x_range = np.array([X.min(), X.max()]).reshape(-1, 1)
    y_pred = model.predict(x_range)
    plt.plot(x_range, y_pred, color='red', linewidth=3, label='Đường hồi quy')
    
    # Thông tin mô hình
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    equation = f'y = {model.weights[0]:.4f}x + {model.bias:.4f}'
    plt.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12)
    
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
    X, y = generate_data(n_samples=10, noise=1.5)
    start_time = time.time()
    # Huấn luyện mô hình
    model = LinearRegression(learning_rate=0.02, n_iterations=1000)
    model.fit(X, y)
    
    # Đánh giá mô hình
    r2_score = model.score(X, y)
    print(f"Hệ số R²: {r2_score:.4f}")
    print(f"Hệ số w: {model.weights[0]:.4f}")
    print(f"Hệ số b: {model.bias:.4f}")
    
    end_time = time.time()  # Lấy thời gian kết thúc
    execution_time = end_time - start_time  # Thời gian thực thi
    print(f"Thời gian thực thi: {execution_time} giây")
    # Vẽ kết quả
    plot_results(X, y, model)
    plot_learning_curve(model)


    x_value = model.learning_rate
    y_value = execution_time

    data = pd.DataFrame({'x': [x_value], 'y': [y_value]})

    # Ghi vào file CSV (append nếu muốn ghi vào cuối file mà không ghi đè)
    data.to_csv('output.csv', mode='a', header=False, index=False)   

    # Đọc dữ liệu từ file CSV
    data = pd.read_csv('output.csv')

    # Kiểm tra dữ liệu đã được đọc thành công
    print(data)

    # Vẽ biểu đồ
    plt.plot(data['x'], data['y'], marker='o', linestyle='-', color='b', label='y = f(x)')

    # Thêm tiêu đề và nhãn cho trục
    plt.title('Biểu đồ dữ liệu Learninng_rate và Time')
    plt.xlabel('Learninng_rate')
    plt.ylabel('Time')

    # Hiển thị biểu đồ
    plt.legend()
    plt.show() 
