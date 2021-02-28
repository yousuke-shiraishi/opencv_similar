import cv2
import glob
import os
import psycopg2
import boto3
import json
from os.path import join, dirname
# from dotenv import load_dotenv
# dotenv_path = join(dirname(__file__), '.env')
# load_dotenv(dotenv_path)
# POSTG_ID = os.environ['PG_ID']
# POSTG_PW = os.environ['PG_PW']
# POSTG_DB = os.environ['PG_DB']
AWS_ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
AWS_STORAGE_BUCKET_NAME = os.environ['AWS_STORAGE_BUCKET_NAME']
AWS_REGION_NAME = os.environ['AWS_REGION_NAME']
DATABASE_URL = os.environ['DATABASE_URL']

SAVE_DIR = "./images"
# カーソル作成
conn = psycopg2.connect(DATABASE_URL, sslmode='require')
# conn = psycopg2.connect(
#     host="0.0.0.0",
#     port=5432,
#     database=POSTG_DB,
#     user=POSTG_ID,
#     password=POSTG_PW)


face_cascade_path = './haarcascade_frontalface_alt.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)
files = glob.glob("./actress/*")
img_size = (200, 200)
# print(files)
for file_n in files:
    src = cv2.imread(file_n, cv2.IMREAD_GRAYSCALE)
    filename = SAVE_DIR + "/" + str(file_n)
    src_gray = src
    faces = face_cascade.detectMultiScale(src_gray)
    for x, y, w, h in faces:
        face_gray = src_gray[y: y + h, x: x + w]

    cur = conn.cursor()
    target_img = cv2.resize(face_gray, img_size)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    detector = cv2.AKAZE_create()
    (_, target_des) = detector.detectAndCompute(target_img, None)
    cv2.imshow("img", face_gray)
    cv2.waitKey(0)
    ans = input("yを押すとデータがDBに入力されます: ")
    if ans == 'y':
        cur.execute("INSERT INTO flask_similar (image_name, image_json_feature) VALUES (%s, %s)",
                    (file_n, json.dumps({file_n: target_des.tolist()})))
        conn.commit()
        cur.close()
    else:
        cur.close()
        continue

    s3 = boto3.resource('s3')  # S3オブジェクトを取得

    bucket = s3.Bucket(AWS_STORAGE_BUCKET_NAME)
    bucket.upload_file(filename, 'actress/' + file_n)
conn.close()
