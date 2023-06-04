import shutil
import time
from typing import Union
import cv2
from typing import List
from fastapi import FastAPI, Request, File, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from util import interpolate, animate
import os
from starlette.background import BackgroundTasks

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/match")
def matchImg():
    img1 = cv2.imread("./images/1.jpg")
    img2 = cv2.imread("./images/2.jpg") 
    orb = cv2.ORB_create(nfeatures=270)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    matchImg = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
    cv2.imwrite("./images/match.jpg", matchImg)
    return FileResponse("./images/match.jpg")

# image -> sequence -> interpolation -> style

async def cleanFiles(_id):
    time.sleep(30)
    os.remove(f"frames{_id}.txt")
    shutil.rmtree(f".\input{_id}")
    shutil.rmtree(f".\output{_id}")

class VideoReq(BaseModel):
    files: List[UploadFile] = File(...)
    filter: str = "anime"

@app.post("/video/{filter}")
async def interpolation_req(filter: str, background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    # print(await request.body())
    # print(files, filter)
    # return "sidkhbf"
    # files, filter = reqBody.files, reqBody.filter
    _id = int(time.time() * 10**8)
    outputFolder = f".\output{_id}"
    inputFolder = f".\input{_id}"
    os.mkdir(inputFolder)
    for file in files:
        with open(f"{inputFolder}\{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    interpolate(inputFolder, outputFolder)
    print(filter)
    if filter == "Anime":
        animate(f"output{_id}")
    # elif filter == "cartoon":
    #     cartoon()
    frame_list = os.listdir(outputFolder)
    with open(f"frames{_id}.txt", "w") as f:
        for frame in frame_list:
            f.write(f"file '{outputFolder}\{frame}'\n")
    os.system(f"ffmpeg -f concat -safe 0 -i frames{_id}.txt -c:v libx264 -vf pad='ceil(iw/2)*2:ceil(ih/2)*2' ./static/output{_id}.mp4")
    background_tasks.add_task(cleanFiles, _id)
    # return FileResponse(f'output{_id}.mp4', media_type='application/octet-stream', filename=f'output{_id}.mp4')
    return {"url":f'/static/output{_id}.mp4'}


# interpolation_req()