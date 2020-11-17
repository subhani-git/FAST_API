# 1. Library imports
import uvicorn
import cv2
import shutil
from fastapi import FastAPI, File, UploadFile
from typing import List
from fastapi.responses import FileResponse
from proper_set_image_detection_code import sample_detection,process_init
import cv2
# 2. Create app and model objects
app = FastAPI()
#def img_detection(img_folder_path, PATH_TO_CKPT, PATH_TO_LABELS, TH, saved_img_path, time_for_inference)
a=[]
@app.post('/FillRequirements/')
async def image(image: List[UploadFile] = File(...)):
    for img in image:
        with open(img.filename, "wb") as buffer:
            shutil.copyfileobj(img.file, buffer)
            a.append(sample_detection(img.filename))
    return a


# @app.post('/FillRequirements/')
# async def image(image: List[UploadFile] = File(...)):
# 	for img in image:
# 		print('111111111')
#         with open(img.filename, "wb") as buffer:
#             shutil.copyfileobj(img.file, buffer)

#             detected_image_path = sample_detection(image.filename)
#         # image_info, img = sample_detection(img)
#       	# json_string = json.dumps(image_info)

#         return detected_image_path


if __name__ == '__main__':
	process_init('./ssd/output_inference_graph.pb/frozen_inference_graph.pb',
	             './inputs/label_map.pbtxt', 1)
	uvicorn.run(app, host='0.0.0.0', port=8000)    



# # 1. Library imports
# import uvicorn
# import cv2
# import shutil
# import json
# from fastapi import FastAPI, File, UploadFile
# import base64
# from io import StringIO
# from typing import List
# from fastapi.responses import FileResponse
# from proper_set_image_detection_code import sample_detection, process_init, base64str_to_PILImage
# import cv2
# # 2. Create app and model objects
# app = FastAPI()
# #def img_detection(img_folder_path, PATH_TO_CKPT, PATH_TO_LABELS, TH, saved_img_path, time_for_inference)
# @app.post('/Fill Requirements/')
# async def image(image: UploadFile = File(...)):
    
    # encoded_string = StringIO(image)
    # img = base64str_to_PILImage(encoded_string)
    # image_info, img = sample_detection(img)

#     #json_string = json.dumps(image_info)
#     return FileResponse(img), image_info
# ​
# if __name__ == '__main__':
#     process_init('./ssd/output_inference_graph.pb/frozen_inference_graph.pb',
#                  './inputs/label_map.pbtxt', 1)
#     uvicorn.run(app, host='127.0.0.1', port=8000)    