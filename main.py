from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import pickle
import keras
from keras.preprocessing.sequence import pad_sequences
import json

app = FastAPI()

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

load_model = keras.models.load_model("model_bidir_lstm.h5")
print(tokenizer)
print(load_model)


sentences = ['is a boy', 'you are good fucker']
for sent in sentences:
    text = pad_sequences(tokenizer.texts_to_sequences([sent]), maxlen=30, padding='pre')
    print(text)
    x = load_model.predict(text)
    print(sent)
    print(x)
# print(tokenizer.texts_to_sequences(['is a boy', 'Fuck you']))
# print(load_model.summary())


html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://localhost:8000/ws");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""


@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        print(data)
        text = pad_sequences(tokenizer.texts_to_sequences([data]), maxlen=30, padding='pre')
        # print(text)
        x = load_model.predict(text)
        resp = json.dumps({'text':data,'confidence':float(x[0][0])})
        # print(resp)
        await websocket.send_text(resp)

