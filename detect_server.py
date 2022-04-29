#!/usr/bin/env python
import requests
from sanic import response
from sanic import Sanic
from PIL import Image
import cv2
import json
from sanic.response import text
import numpy as np
from app.process_module.image_detect_module import YoloV5DetectModule
from sanic.constants import HTTPMethod

app = Sanic("detect_server")

detect_module = YoloV5DetectModule(skippable=False)
names = detect_module.names


@app.route("/")
async def test(request):
    return text('Hello World!')


@app.route("get_detect_result", methods=['POST'])
async def get_detect_result(request):
    if request.method != 'POST':
        return response.json([])

    return response.json(
        {
            "pred":
                [detect_module.detect(
                    np.array(json.loads(request.form['data'][0])).astype("uint8")
                )[0].tolist()],
            "names": names
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
