import json
from fastapi import FastAPI, Response
from pydantic import BaseModel
from fastapi.responses import UJSONResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import SpeechToText as ST
import SpeechToText_cloud as stc
app = FastAPI()
#CORS Error
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class STT(BaseModel):
    file_name: str

class List_stt(BaseModel):
    file_name: list

@app.get("/")
async def index():
    return {"message": "Hello World from Speech to text "}


@app.post('/transcribe_audio', response_class=UJSONResponse)
def process_text(st: STT):
    #print("transcribe_audio called ")
    speech_to_text = ST.transcribe_audio(st.file_name)
    final_json = json.dumps(speech_to_text, indent=4, default=str, ensure_ascii=False)
    return Response(content=final_json, media_type='application/json')


@app.post('/transcribe_audio_cloud', response_class=UJSONResponse)
def process_text(st: STT):
    #print("transcribe_audio called ")
    speech_to_text = stc.transcribe_audio_with_speech_recognition(st.file_name)
    final_json = json.dumps(speech_to_text, indent=4, default=str, ensure_ascii=False)
    return Response(content=final_json, media_type='application/json')

@app.post('/transcribe_audio_files', response_class=UJSONResponse)
def transcribe_audio_files(st: List_stt):
    speech_to_text = []
    for file in st.file_name:

    #print("transcribe_audio called ")
        trans = ST.transcribe_audio(file)
        speech_to_text.append(trans)
    final_json = json.dumps(speech_to_text, indent=4, default=str, ensure_ascii=False)

    return Response(content=final_json, media_type='application/json')


@app.post('/transcribe_audio_large_files', response_class=UJSONResponse)
def transcribe_audio_large_files(st: STT):
    speech_to_text = ST.transcribe_audio_largefile(st.file_name)
    final_json = json.dumps(speech_to_text, indent=4, default=str, ensure_ascii=False)
    return Response(content=final_json, media_type='application/json')



"""@app.post('/convert_bin_to_wav')
async def save_file(file: UploadFile):
    try:
        file_path = file.filename
        async with aiofiles.open(file_path, "wb") as f:
            contents = await file.read()
            await f.write(contents)

        # Perform any additional processing or actions here
        call_base_api(file_path)
        return "File uploaded and processed successfully"

    except Exception as e:
        return str(e)"""


# if __name__ == "__main__":
#     uvicorn.run("SpeechToTextFastAPI:app", host="127.0.0.1", port=5000, reload=True)
