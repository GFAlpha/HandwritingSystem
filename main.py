import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from PIL import Image
import pytesseract
import joblib

# ===== 设置 Tesseract路径 =====
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ===== 全局配置 =====
IMAGE_SIZE = (28, 28)
DATASET_PATH = "data"  # 数据集路径
MODEL_SAVE_PATH = "models"  # 模型保存路径

if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

# ===== 图像预处理函数 =====
"""
    对输入图像进行灰度化、尺寸调整和二值化处理    
        Args:
            image_path (str): 输入图像的完整文件路径
    
        Returns:
            numpy.ndarray: 预处理后的二值化图像矩阵
        
        Raises:
            FileNotFoundError: 当图像路径不存在时引发异常
"""
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"找不到图片: {image_path}")
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)
    return img

# ===== 提取 HOG 特征 =====
"""
    使用方向梯度直方图（HOG）算法提取图像特征
        Args:
            image (numpy.ndarray): 预处理后的灰度图像矩阵
        
        Returns:
            numpy.ndarray: 提取的特征向量
"""
def extract_hog_features(image):
    features = hog(image, pixels_per_cell=(7,7), cells_per_block=(2,2),
                   block_norm='L2-Hys', visualize=False)
    return features

# ===== 加载数据集 =====
"""
    加载并处理数据集目录下的所有PNG图像文件
    
    文件命名格式要求：
    - 格式：数字标签_用户标签.png（例：5_userA.png）
    - 自动跳过不符合命名规范的文件
    
    Returns:
        tuple: 包含四个元素的元组
            - X_digit (numpy.ndarray): 数字特征数据集
            - y_digit (numpy.ndarray): 数字标签数组 
            - X_user (numpy.ndarray): 用户特征数据集
            - y_user (numpy.ndarray): 用户标签数组
"""
def load_digits_data():
    X_digit, y_digit = [], []
    X_user, y_user = [], []
    
    for filename in os.listdir(DATASET_PATH):
        if not filename.endswith(".png"):
            continue
            
        parts = filename.split("_")
        if len(parts) < 2:
            continue
            
        digit_label = int(parts[0])  
        user_label = parts[1]        
        
        image_path = os.path.join(DATASET_PATH, filename)
        try:
            image = preprocess_image(image_path)
            features = extract_hog_features(image)
            
            # 数字识别数据
            X_digit.append(features)
            y_digit.append(digit_label)
            
            # 用户识别数据
            X_user.append(features)
            y_user.append(user_label)
            
        except Exception as e:
            print(f"加载失败: {filename}, 错误: {e}")
    
    return (
        np.array(X_digit), np.array(y_digit),
        np.array(X_user), np.array(y_user)
    )

# ===== 训练数字识别模型 =====
"""
    训练并保存数字识别SVM分类器
    
    流程说明：
    1. 加载预处理后的数字数据集
    2. 按8:2比例分割训练集/测试集
    3. 使用RBF核的SVM进行训练
    4. 评估模型准确率
    5. 持久化保存模型到文件

"""
def train_digit_recognition_model():
    X_digit, y_digit, _, _ = load_digits_data()
    X_train, X_test, y_train, y_test = train_test_split(X_digit, y_digit, test_size=0.2, random_state=42)
    
    model = svm.SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"数字识别模型准确率: {acc*100:.2f}%")
    
    joblib.dump(model, os.path.join(MODEL_SAVE_PATH, 'digit_model.pkl'))

# ===== 训练用户识别模型 =====
"""
    训练并保存用户识别SVM分类器
    
    流程说明：
    1. 加载预处理后的用户数据集
    2. 按8:2比例分割训练集/测试集
    3. 使用RBF核的SVM进行训练
    4. 评估模型准确率
    5. 持久化保存模型到文件
    
    注意：
    使用与数字识别相同的特征数据，但采用用户标签进行分类

"""
def train_user_recognition_model():
    _, _, X_user, y_user = load_digits_data()
    X_train, X_test, y_train, y_test = train_test_split(X_user, y_user, test_size=0.2, random_state=42)
    
    model = svm.SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"用户识别模型准确率: {acc*100:.2f}%")
    
    joblib.dump(model, os.path.join(MODEL_SAVE_PATH, 'user_model.pkl'))

# ===== 单数字识别并输出匹配概率值 =====
"""
    对单个数字图像进行识别并返回置信度
    
    Args:
        image_path (str): 待识别图像路径
        digit_model (sklearn.svm.SVC): 已训练的数字分类模型
        
    Returns:
        tuple: (预测结果字符串, 置信度浮点数)
"""
def predict_single_digit_with_confidence(image_path, digit_model):
    try:
        image = preprocess_image(image_path)
        features = extract_hog_features(image).reshape(1, -1)
        prediction = digit_model.predict(features)[0]
        probabilities = digit_model.predict_proba(features)[0]
        confidence = probabilities[prediction]
        return str(prediction), float(confidence)
    except Exception as e:
        print("单数字识别错误:", e)
        return "?", 0.0

# ===== 用户识别并输出匹配概率值 =====
"""
    识别图像书写用户并返回置信度
    
    Args:
        image_path (str): 包含用户笔迹的图像路径
        user_model (sklearn.svm.SVC): 用户分类模型
        
    Returns:
        tuple: (用户标识字符串, 置信度浮点数)
"""
def predict_user_with_confidence(image_path, user_model):
    try:
        image = preprocess_image(image_path)
        features = extract_hog_features(image).reshape(1, -1)
        prediction = user_model.predict(features)[0]
        probabilities = user_model.predict_proba(features)[0]
        confidence = probabilities.max()
        return prediction, float(confidence)
    except Exception as e:
        print("用户识别错误:", e)
        return "?", 0.0

# ===== 数字识别: OCR辅助分割 =====
"""
    使用Tesseract OCR引擎识别多数字图像
    
    配置参数说明：
    - psm 7: 单行文本识别模式
    - 白名单限制为数字字符
    
    Args:
        image_path (str): 包含多数字的图像路径
        
    Returns:
        str: 识别结果字符串（可能为空）
"""
def recognize_multiple_digits(image_path):
    try:
        text = pytesseract.image_to_string(Image.open(image_path),
                                          config='--psm 7 -c tessedit_char_whitelist=0123456789')
        return text.strip()
    except Exception as e:
        print("Tesseract OCR出错:", e)
        return ""

# ===== 预测数字主函数并输出匹配概率值 =====
"""
    综合识别流程入口函数
    
    执行流程：
    1. 加载两个预训练模型
    2. 尝试OCR识别多个数字
    3. 并行执行用户身份识别
    4. 根据OCR结果选择返回方式
    
    Args:
        image_path (str): 待识别图像路径
        
    Returns:
        tuple: (数字识别结果, 用户识别结果)
"""
def predict_number_with_confidence_and_user(image_path):
    try:
        digit_model = joblib.load(os.path.join(MODEL_SAVE_PATH, 'digit_model.pkl'))
        user_model = joblib.load(os.path.join(MODEL_SAVE_PATH, 'user_model.pkl'))
        
        # 尝试使用OCR识别多个数字
        ocr_result = recognize_multiple_digits(image_path)
        
        # 无论是否OCR成功都进行用户识别
        user_id, user_conf = predict_user_with_confidence(image_path, user_model)
        
        if ocr_result:
            return (ocr_result, user_id)
        else:
            digit, digit_conf = predict_single_digit_with_confidence(image_path, digit_model)
            return (digit, user_id)
        
    except Exception as e:
        return (f"识别错误: {e}", "?")

# ===== 主菜单 =====
"""显示系统命令行交互菜单"""
def main_menu():
    print("\n===== 手写数字识别系统 v6.3 =====")
    print("1. 训练数字识别模型")
    print("2. 训练用户识别模型")
    print("3. 预测数字及用户")
    print("4. 退出系统")

# ===== 系统运行入口 =====
""""系统主控制循环"""
def main():
    while True:
        main_menu()
        choice = input("请选择操作: ").strip()
        
        if choice == "1":
            print("开始训练数字识别模型...")
            train_digit_recognition_model()
            
        elif choice == "2":
            print("开始训练用户识别模型...")
            train_user_recognition_model()
            
        elif choice == "3":
            image_path = input("输入待识别图片路径: ").strip()
            if not os.path.exists(image_path):
                print("图片路径无效!")
                continue
            
            result = predict_number_with_confidence_and_user(image_path)
            
            if isinstance(result, tuple) and len(result) == 2:
                number, user = result
                print(f"\n识别结果: {number}")
                if user != "?":
                    print(f"最可能的书写人: {user}")
                else:
                    print("无法确定书写人")
            else:
                print(f"\n识别结果: {result}")
                
        elif choice == "4":
            print("退出系统。")
            break
            
        else:
            print("无效选项，请重新选择。")

if __name__ == "__main__":
    main()