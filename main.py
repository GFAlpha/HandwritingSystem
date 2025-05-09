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

# ===== 设置 Tesseract路径(必须修改为你的实际安装路径) =====
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ===== 全局配置 =====
IMAGE_SIZE = (28, 28)
DATASET_PATH = "data"  # 所有图片都在这个文件夹里
MODEL_SAVE_PATH = "models"

if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

# ===== 图像预处理函数 =====
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"找不到图片: {image_path}")
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)
    return img

# ===== 提取 HOG 特征 =====
def extract_hog_features(image):
    features = hog(image, pixels_per_cell=(7,7), cells_per_block=(2,2),
                   block_norm='L2-Hys', visualize=False)
    return features

# ===== 加载数据集 =====
def load_digits_data():
    X_digit, y_digit = [], []
    X_user, y_user = [], []
    
    for filename in os.listdir(DATASET_PATH):
        if not filename.endswith(".png"):
            continue
            
        parts = filename.split("_")
        if len(parts) < 2:
            continue
            
        digit_label = int(parts[0])  # 如"6"
        user_label = parts[1]        # 如"user01"
        
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
def recognize_multiple_digits(image_path):
    try:
        text = pytesseract.image_to_string(Image.open(image_path),
                                          config='--psm 7 -c tessedit_char_whitelist=0123456789')
        return text.strip()
    except Exception as e:
        print("Tesseract OCR出错:", e)
        return ""

# ===== 预测数字主函数并输出匹配概率值 =====
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
def main_menu():
    print("\n===== 手写数字识别系统 v6.3 =====")
    print("1. 训练数字识别模型")
    print("2. 训练用户识别模型")
    print("3. 预测数字及用户")
    print("4. 退出系统")

# ===== 系统运行入口 =====
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