import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os


def load_image(image_bytes):
    img = Image.open(image_bytes)
    preprocess = transforms.Compose([
        transforms.Resize(256),  # リサイズ
        transforms.CenterCrop(224),  # 中心をクロップ
        transforms.ToTensor(),  # テンソルへの変換
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 正規化
    ])
    img_tensor = preprocess(img)
    img_tensor.unsqueeze_(0)
    return img_tensor


def predict(model, img_tensor, class_names, top_k=3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.topk(outputs, top_k, dim=1)
        probabilities = torch.softmax(outputs, dim=1)
        top_probabilities, top_indices = torch.topk(probabilities, top_k, dim=1)

    return [(class_names[index], float(prob)) for index, prob in zip(top_indices[0], top_probabilities[0])]


# クラス名の定義
class_names = [
        "健全",
        "うどんこ病",
        "灰色かび病",
        "炭疽病",
        "べと病",
        "褐斑病",
        "つる枯病",
        "斑点細菌病",
        "CCYV",
        "モザイク病",
        "MYSV",
    ]

model_path = './model_cu.pth'

# モデルの読み込み
model_ft = models.resnet18()
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

# 学習済みモデルのパラメータをロード
model_ft.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model_ft.eval()

# Streamlitアプリの構築
st.title('きゅうりの病害判別アプリ')

uploaded_file = st.sidebar.file_uploader("画像ファイルをアップロードしてください", type=['jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='アップロードされた画像', use_column_width=True)
    img_tensor = load_image(uploaded_file)
    predictions = predict(model_ft, img_tensor, class_names, top_k=3)

    st.header('予測結果:')
    for breed, prob in predictions:
        st.write(f"{breed}: {prob * 100:.2f}%")
