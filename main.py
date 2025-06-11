import numpy as np
import matplotlib.pyplot as plt

# 1. 目的関数とその導関数を定義
def objective_function(x):
    """最小化したい関数: f(x) = (x-3)^2 + 1"""
    return (x - 3)**2 + 1

def gradient(x):
    """目的関数の導関数: f'(x) = 2(x-3)"""
    return 2 * (x - 3)

# 2. 勾配降下法の実装
def gradient_descent(start_x, learning_rate, max_iterations, tolerance=1e-6):
    """
    勾配降下法で最小値を求める
    
    Parameters:
    - start_x: 初期値
    - learning_rate: 学習率
    - max_iterations: 最大反復回数
    - tolerance: 収束判定の閾値
    
    Returns:
    - x_history: xの履歴
    - f_history: 関数値の履歴
    """
    x = start_x
    x_history = [x]
    f_history = [objective_function(x)]
    
    for i in range(max_iterations):
        # 現在の点での勾配を計算
        grad = gradient(x)
        
        # パラメータを更新
        x_new = x - learning_rate * grad
        
        # 履歴を記録
        x_history.append(x_new)
        f_history.append(objective_function(x_new))
        
        # 収束判定
        if abs(x_new - x) < tolerance:
            print(f"収束しました（反復回数: {i+1}）")
            break
        
        x = x_new
    
    return np.array(x_history), np.array(f_history)

# 3. 実行とビジュアライゼーション
def visualize_gradient_descent():
    # パラメータ設定
    start_x = 0.0
    learning_rate = 0.1
    max_iterations = 20
    
    print("=== 勾配降下法による最小化 ===")
    print(f"目的関数: f(x) = (x-3)² + 1")
    print(f"初期値: {start_x}")
    print(f"学習率: {learning_rate}")
    print("-" * 40)
    
    # 勾配降下法を実行
    x_history, f_history = gradient_descent(start_x, learning_rate, max_iterations)
    
    # 結果表示
    print(f"最終結果: x = {x_history[-1]:.6f}, f(x) = {f_history[-1]:.6f}")
    print(f"理論値: x = 3.0, f(x) = 1.0")
    
    # プロット用データ
    x_plot = np.linspace(-1, 6, 1000)
    y_plot = objective_function(x_plot)
    
    # 図を作成
    plt.figure(figsize=(12, 8))
    
    # 上のサブプロット: 関数と最適化過程
    plt.subplot(2, 1, 1)
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x) = (x-3)² + 1')
    plt.plot(x_history, f_history, 'ro-', markersize=8, linewidth=2, 
             label='勾配降下法の軌跡')
    plt.plot(x_history[0], f_history[0], 'go', markersize=12, label='開始点')
    plt.plot(x_history[-1], f_history[-1], 'ko', markersize=12, label='終了点')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('勾配降下法による最小化過程')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 下のサブプロット: 収束過程
    plt.subplot(2, 1, 2)
    iterations = range(len(f_history))
    plt.plot(iterations, f_history, 'bo-', linewidth=2, markersize=6)
    plt.axhline(y=1.0, color='r', linestyle='--', label='理論最小値 = 1.0')
    plt.xlabel('反復回数')
    plt.ylabel('関数値 f(x)')
    plt.title('収束過程')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 各ステップの詳細を表示
    print("\n=== 各ステップの詳細 ===")
    for i in range(min(10, len(x_history))):  # 最初の10ステップまで表示
        print(f"Step {i}: x = {x_history[i]:.6f}, f(x) = {f_history[i]:.6f}, "
              f"gradient = {gradient(x_history[i]):.6f}")

# 4. 異なる学習率での比較
def compare_learning_rates():
    """異なる学習率での収束速度を比較"""
    learning_rates = [0.01, 0.1, 0.3, 0.9]
    start_x = 0.0
    max_iterations = 30
    
    plt.figure(figsize=(12, 8))
    
    for i, lr in enumerate(learning_rates):
        x_history, f_history = gradient_descent(start_x, lr, max_iterations)
        
        plt.subplot(2, 2, i+1)
        iterations = range(len(f_history))
        plt.plot(iterations, f_history, 'bo-', linewidth=2, markersize=4)
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
        plt.title(f'学習率 = {lr}')
        plt.xlabel('反復回数')
        plt.ylabel('f(x)')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # 対数スケール
    
    plt.tight_layout()
    plt.show()

# 実行
if __name__ == "__main__":
    # 基本的な勾配降下法の実行
    visualize_gradient_descent()
    
    print("\n" + "="*50)
    print("異なる学習率での比較")
    print("="*50)
    
    # 学習率比較
    compare_learning_rates()
