import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_confusion_matrix(cm, classes, title='混淆矩阵', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')

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
    
    # 绘制混淆矩阵图
    plot_confusion_matrix(cm, ['低浓度', '中浓度', '高浓度'], title='SVM分类混淆矩阵')
    plt.savefig('confusion_matrix.png')
    print("\n混淆矩阵图已保存为 confusion_matrix.png")
    
    # 可视化特征分布（取前两个特征）
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.legend(handles=scatter.legend_elements()[0], labels=['低浓度', '中浓度', '高浓度'])
    plt.xlabel('传感器1 (标准化)')
    plt.ylabel('传感器2 (标准化)')
    plt.title('三种浓度传感器数据分布')
    plt.savefig('sensor_data_distribution1.png')
    print("\n特征分布图已保存为 sensor_data_distribution1.png")

if __name__ == '__main__':
    main()
