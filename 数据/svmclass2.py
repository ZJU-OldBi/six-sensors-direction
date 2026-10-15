import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sensor_names = ['传感器1', '传感器2', '传感器3', '传感器4', '传感器5']

def load_data(file_path, label):
    """加载数据文件，提取中间5分钟的数据（气体加入阶段）"""
    df = pd.read_csv(file_path, encoding='gbk')
    
    total_samples = len(df)
    print(f"文件 {file_path} 总样本数: {total_samples}")
    
    # 数据结构：2分钟静息 + 5分钟气体 + 2分钟释放 = 9分钟
    # 采样频率约为100Hz（从时间戳看约12ms一个样本）
    # 9分钟 = 540秒，约 540 * 100 = 54000 个样本
    # 取中间5/9的数据作为气体加入阶段
    
    rest_ratio = 2/9
    gas_ratio = 5/9
    
    start_idx = int(total_samples * rest_ratio)
    end_idx = int(total_samples * (rest_ratio + gas_ratio))
    
    gas_data = df.iloc[start_idx:end_idx]
    print(f"提取中间5分钟数据: {len(gas_data)} 个样本")
    
    # 使用5个传感器数据作为特征
    features = gas_data[['传感器1', '传感器2', '传感器3', '传感器4', '传感器5']].values
    labels = np.full(len(features), label)
    
    return features, labels

def main():
    # 文件路径
    low_file = r'乙醇低sensor_data_2026-04-21_19-59-32.csv'
    medium_file = r'乙醇中sensor_data_2026-04-21_20-11-09.csv'
    high_file = r'乙醇高sensor_data_2026-04-21_20-20-36.csv'
    
    # 加载三种浓度的数据
    print("=== 加载低浓度数据 ===")
    low_features, low_labels = load_data(low_file, 0)
    
    print("\n=== 加载中浓度数据 ===")
    medium_features, medium_labels = load_data(medium_file, 1)
    
    print("\n=== 加载高浓度数据 ===")
    high_features, high_labels = load_data(high_file, 2)
    
    # 合并数据
    X = np.vstack([low_features, medium_features, high_features])
    y = np.hstack([low_labels, medium_labels, high_labels])
    
    print(f"\n总数据集: {len(X)} 个样本")
    print(f"类别分布: 低浓度={sum(y==0)}, 中浓度={sum(y==1)}, 高浓度={sum(y==2)}")
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集 (7:3)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\n训练集: {len(X_train)} 个样本")
    print(f"测试集: {len(X_test)} 个样本")
    
    # 创建SVM分类器
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    
    # 训练模型
    print("\n=== 训练SVM模型 ===")
    svm.fit(X_train, y_train)
    
    # 预测
    y_pred = svm.predict(X_test)
    
    # 评估
    print("\n=== 模型评估 ===")
    print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['低浓度', '中浓度', '高浓度']))
    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # 可视化特征分布（所有5个传感器的散点图矩阵）
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    colors = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']
    
    for i in range(5):
        for j in range(5):
            if i == j:
                for cls in range(3):
                    axes[i, j].hist(X_scaled[y == cls, i], bins=30, alpha=0.5, color=colors[cls], label=['低浓度', '中浓度', '高浓度'][cls])
                axes[i, j].set_title(sensor_names[i])
            else:
                for cls in range(3):
                    axes[i, j].scatter(X_scaled[y == cls, j], X_scaled[y == cls, i], alpha=0.5, color=colors[cls], s=10)
            if j == 0 and i != 0:
                axes[i, j].set_ylabel(sensor_names[i])
            if i == 4:
                axes[i, j].set_xlabel(sensor_names[j])
    
    handles = [plt.Line2D([0], [0], marker='o', color=c, linestyle='', markersize=8) for c in colors[:3]]
    fig.legend(handles, ['低浓度', '中浓度', '高浓度'], loc='upper right', bbox_to_anchor=(1.15, 0.95))
    plt.tight_layout()
    plt.savefig('sensor_data_distribution.png', bbox_inches='tight')
    print("\n特征分布图已保存为 sensor_data_distribution.png")
    
    # 绘制每个传感器的数据分布图
    fig, axes = plt.subplots(5, 1, figsize=(10, 15))
    for i in range(5):
        for cls in range(3):
            axes[i].hist(X_scaled[y == cls, i], bins=50, alpha=0.5, color=colors[cls], label=['低浓度', '中浓度', '高浓度'][cls])
        axes[i].set_xlabel(sensor_names[i] + ' (标准化)')
        axes[i].set_ylabel('样本数')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig('sensor_histograms.png')
    print("传感器直方图已保存为 sensor_histograms.png")

if __name__ == '__main__':
    main()
