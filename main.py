# from fastapi import FastAPI
# from fastapi.responses import FileResponse
# from typing import Optional
# app = FastAPI()
#
#
# # @app.get("/")
# # async def root():
# #     return {"message": "Hello World"}
# #
# # @app.get("/component/{component_id}")   #path parameter
# # async def componenet(component_id: int):
# #     return {'component_id': component_id}
# #
# # @app.get("/component/")
# # async def read_comment(id: int, text: Optional[str]):
# #     return {'id': id, 'text': text}
# @app.get("/")
# async def main():
#     yield FileResponse("D:\\PycharmProjects\\fastapi\\result\\cam2.jpg")
#
# a= next(main

from typing import List
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

app = FastAPI()


@app.post("/files/")
async def create_files(files: List[bytes] = File(...)):
    return {"file_sizes": [len(file) for file in files]}


@app.post("/uploadfiles/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    return {"filenames": [file.filename for file in files]}


@app.get("/")
async def main():
    content = """
<body>
<form action="/files/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)
# if __name__ == '__main__':
# #     uvicorn.run(app, host='127.0.0.0', port=8000)