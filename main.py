import json
from fastapi import FastAPI, WebSocket,WebSocketDisconnect,BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pickle
from starlette.requests import Request
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

progress = Progress(
    TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    DownloadColumn(),
    "•",
    TransferSpeedColumn(),
    "•",
    TimeRemainingColumn(),
)

cred = credentials.Certificate("cred.json")
firebase_admin.initialize_app(cred,{"storageBucket":"ybullyy.appspot.com"})


db = firestore.client()
bucket = storage.bucket() 
models_ref = db.collection(u'models')
transaction = db.transaction()


CONTRACTION_MAP = {"ain't": 'is not', "aren't": 'are not', "can't": 'cannot', "can't've": 'cannot have', "'cause": 'because', "could've": 'could have', "couldn't": 'could not', "couldn't've": 'could not have', "didn't": 'did not', "doesn't": 'does not', "don't": 'do not', "hadn't": 'had not', "hadn't've": 'had not have', "hasn't": 'has not', "haven't": 'have not', "he'd": 'he would', "he'd've": 'he would have', "he'll": 'he will', "he'll've": 'he he will have', "he's": 'he is', "how'd": 'how did', "how'd'y": 'how do you', "how'll": 'how will', "how's": 'how is', "I'd": 'I would', "I'd've": 'I would have', "I'll": 'I will', "I'll've": 'I will have', "I'm": 'I am', "I've": 'I have', "i'd": 'i would', "i'd've": 'i would have', "i'll": 'i will', "i'll've": 'i will have', "i'm": 'i am', "i've": 'i have', "isn't": 'is not', "it'd": 'it would', "it'd've": 'it would have', "it'll": 'it will', "it'll've": 'it will have', "it's": 'it is', "let's": 'let us', "ma'am": 'madam', "mayn't": 'may not', "might've": 'might have', "mightn't": 'might not', "mightn't've": 'might not have', "must've": 'must have', "mustn't": 'must not', "mustn't've": 'must not have', "needn't": 'need not', "needn't've": 'need not have', "o'clock": 'of the clock', "oughtn't": 'ought not', "oughtn't've": 'ought not have', "shan't": 'shall not', "sha'n't": 'shall not', "shan't've": 'shall not have', "she'd": 'she would', "she'd've": 'she would have', "she'll": 'she will', "she'll've": 'she will have',
    "she's": 'she is', "should've": 'should have', "shouldn't": 'should not', "shouldn't've": 'should not have', "so've": 'so have', "so's": 'so as', "that'd": 'that would', "that'd've": 'that would have', "that's": 'that is', "there'd": 'there would', "there'd've": 'there would have', "there's": 'there is', "they'd": 'they would', "they'd've": 'they would have', "they'll": 'they will', "they'll've": 'they will have', "they're": 'they are', "they've": 'they have', "to've": 'to have', "wasn't": 'was not', "we'd": 'we would', "we'd've": 'we would have', "we'll": 'we will', "we'll've": 'we will have', "we're": 'we are', "we've": 'we have', "weren't": 'were not', "what'll": 'what will', "what'll've": 'what will have', "what're": 'what are', "what's": 'what is', "what've": 'what have', "when's": 'when is', "when've": 'when have', "where'd": 'where did', "where's": 'where is', "where've": 'where have', "who'll": 'who will', "who'll've": 'who will have', "who's": 'who is', "who've": 'who have', "why's": 'why is', "why've": 'why have', "will've": 'will have', "won't": 'will not', "won't've": 'will not have', "would've": 'would have', "wouldn't": 'would not', "wouldn't've": 'would not have', "y'all": 'you all', "y'all'd": 'you all would', "y'all'd've": 'you all would have', "y'all're": 'you all are', "y'all've": 'you all have', "you'd": 'you would', "you'd've": 'you would have', "you'll": 'you will', "you'll've": 'you will have', "you're": 'you are', "you've": 'you have'}



def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    text = text.split()
    for i in range(len(text)):
        word = text[i]
        if word in contraction_mapping:
            text[i] = contraction_mapping[word]
    text = " ".join(text)
    text = text.replace("'s",'')
    return text

def preprocess(data):
    new_list = []
    for text in data:
        text = text.lower()
        clean_text = re.sub(r'[^a-zA-Z0-9. ]','',expand_contractions(text))
        new_list.append(clean_text)
    final_text = pad_sequences(tokenizer.texts_to_sequences(new_list),maxlen = 30, padding='pre')
    return final_text


with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = keras.models.load_model("model.h5")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def get():
    return { 'message': 'Welcome to yBully api !' }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            preprocessed_data = preprocess([data])
            predictions = model.predict(preprocessed_data)
            predictions = predictions.tolist()

            resp = json.dumps({'text':data,'confidence':float(predictions[0][0])})
            print(resp)
            await websocket.send_text(resp)
    except WebSocketDisconnect:
        print("Client Disconnected")


@firestore.transactional
def update_active_model(transaction,model_ref,model_version):
    doc = model_ref.where('isActive','==',True).limit(1).get()
    data = doc[0].to_dict()
    active_model_ref = model_ref.document(str(data['version']))
    new_model_ref = model_ref.document(str(model_version))
    transaction.update(active_model_ref,{'isActive':False})
    transaction.update(new_model_ref,{'isActive':True})

SCOPES = ['https://www.googleapis.com/auth/drive']

credentials = service_account.Credentials.from_service_account_file("./service_account.json",scopes=SCOPES)
drive = build('drive', 'v3', credentials=credentials)

def update_model(download_url,model_version):
    file_name = "model.h5"
    request = drive.files().get_media(fileId=download_url,supportsAllDrives=True)
    with open(file_name,"wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
    global model
    try:
        update_active_model(transaction,models_ref,model_version)
        new_model = keras.models.load_model("model.h5")
        model = new_model
        print("Model updated successfully")
    except Exception as e:
        print(e)
        print("Model update failed")


@app.post("/update")
async def update(req:Request,background_task:BackgroundTasks):
    # Data format to be sent:
    # {
    #     "download_url":"**",
    #     "model_version":"**"
    # }
    body = await req.json()
    url = body['download_url']
    model_version = body['model_version']

    background_task.add_task(update_model,url,model_version)
    return {'message':'Model will be updated shortly'}
    
