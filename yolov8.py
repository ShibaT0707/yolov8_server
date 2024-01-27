from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import os
import mysql.connector
from flask_cors import CORS 
import shutil

app = Flask(__name__)
CORS(app)
model = YOLO("best.pt")

mapp = {}
answer = {}

foods = [
    "ポテトサラダ",
    "ハヤシライス",
    "かまぼこ",
    "牛丼",
    "いなりずし",
    "アジの開き",
    "クリームシチュー",
    "さんまの竜田揚げ",
    "春巻き",
    "かき揚げ丼",
    "カツ丼",
    "しょうゆラーメン",
    "カニクリームコロッケ",
    "チャーハン",
    "カツサンド",
    "コーンバター",
    "アジ刺身",
    "かけそば",
    "とんかつ",
    "ホットドッグ",
    "ナポリタン",
    "きつねうどん",
    "うな丼",
    "ぎょうざ",
    "カレー",
    "カルボナーラ",
    "きつねそば",
    "ご飯",
    "アジフライ",
    "トースト",
    "しゅうまい",
    "ホットケーキ",
    "サンドイッチ",
    "大学芋",
    "チキン南蛮",
    "ハンバーグ",
    "チキンナゲット",
    "たこ焼き",
    "ちゃんぽん",
    "カレーうどん",
    "アジのマリネ",
    "オムライス",
    "かけうどん",
    "お好み焼き"
]

# MySQL データベースに接続
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="insucount"
)

cursor = db.cursor()

@app.route('/predict', methods=['POST'])
def predict_image():
    try:
        # 画像をPOSTから受け取る
        image = request.files['image']
        image.save("uploaded_image.jpg")

        # 画像をYOLOで処理
        with open("uploaded_image.jpg", "rb") as f:
            image = Image.open(BytesIO(f.read()))

        model.predict(image, save_txt=True, save_conf=True)

        new_folder_path = os.path.join("runs/detect/predict/")
        new_labels_path = os.path.join(new_folder_path, "labels.txt")

        with open(new_labels_path, "r") as file:
            lines = file.readlines()
            first_numbers = [int(item.split()[0]) for item in lines]

            for i in range(len(first_numbers)):
                food_index = first_numbers[i]
                answer[i] = {"food": foods[food_index]}

                # MySQL データベースから炭水化物値を取得
                query = f"SELECT Carbo FROM carbohydrate WHERE FoodName = '{foods[food_index]}'"
                cursor.execute(query)
                carbo_result = cursor.fetchone()
                if carbo_result:
                    answer[i]["carbo"] = carbo_result[0]
                else:
                    answer[i]["carbo"] = "Not available"

        
        print(answer)

        # ディレクトリを削除
        shutil.rmtree("runs")

        # YOLOの結果をJSONに変換
        return jsonify(answer)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
