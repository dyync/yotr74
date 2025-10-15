import numpy as np
import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms

import uvicorn
import redis.asyncio as redis
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from datetime import datetime
import os
import logging
import re
from datetime import datetime




norm_layer = nn.InstanceNorm2d

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features)
                        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()

        # Initial convolution block
        model0 = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    norm_layer(64),
                    nn.ReLU(inplace=True) ]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model1 += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features//2
        for _ in range(2):
            model3 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [  nn.ReflectionPad2d(3),
                        nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out

model1 = Generator(3, 1, 3)
model1.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model1.eval()

model2 = Generator(3, 1, 3)
model2.load_state_dict(torch.load('model2.pth', map_location=torch.device('cpu')))
model2.eval()

import os
import uuid

def predict(input_img, ver):
    print(f'[/predict] input_img: {input_img}')
    print(f'[/predict] ver      : {ver}')
    input_img = Image.open(input_img)
    transform = transforms.Compose([
        transforms.Resize((1536, 1536), Image.BICUBIC),
        transforms.ToTensor()
    ])
    input_img = transform(input_img)
    input_img = torch.unsqueeze(input_img, 0)

    drawing = 0
    with torch.no_grad():
        if ver == 'Simple Lines':
            drawing = model2(input_img)[0].detach()
        else:
            drawing = model1(input_img)[0].detach()
    
    drawing = transforms.ToPILImage()(drawing)
    print(f'[/predict] drawing      : {drawing}')
    
    # Save image to file and return file path
    output_path = f"output_{uuid.uuid4().hex[:8]}.png"
    drawing.save(output_path)
    
    return output_path




def prompt_to_filename(prompt, extension="png", add_timestamp=False):
    filename = prompt.lower()
    filename = re.sub(r'[^\w\s-]', '', filename)
    filename = re.sub(r'[-\s]+', '_', filename)
    filename = filename.strip('_')

    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{timestamp}"
    
    return f"{filename}.{extension}"

redis_connection = None

def start_redis(req_redis_port):
    try:
        r = redis.Redis(host="redis", port=req_redis_port, db=0)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [start_redis] Redis started successfully.')
        return r
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [START] [start_redis] Failed to start Redis on port {req_redis_port}: {e}')
        raise

app = FastAPI()

@app.get("/")
async def root():
    return 'Hello from image generation server!'

@app.get("/vipimages/{image_name}")
async def get_image(image_name: str):
    image_path = f"./{image_name}"
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    if not image_name.lower().endswith('.png'):
        raise HTTPException(status_code=400, detail="Only PNG files are supported")
    return FileResponse(image_path, media_type="image/png")

@app.post("/generate")
async def generate(request: Request):
    try:
        req_data = await request.json()
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate] req_data > {req_data}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate] req_data > {req_data}')
        
        if req_data["method"] == "status":
            return JSONResponse({"result_status": 200, "result_data": "ok"})
            
        if req_data["method"] == "generate_image":
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] trying to generate image...')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] trying to generate image...')


            req_image_path = prompt_to_filename(req_data["image_prompt"])

            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] req_image_path: {req_image_path}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] req_image_path: {req_image_path}')

            
            result = predict(
                req_image_path,
                req_data["req_ver"]
            )
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] result {result}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [generate_image] result {result}')
            
            return JSONResponse({"result_status": 200, "result_data": result})            
        
        if req_data["method"] == "go":
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [go] trying to generate image...')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [go] trying to generate image...')


            req_image_path = prompt_to_filename(req_data["image_prompt"])

            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [go] req_data["req_path"]: {req_data["req_path"]}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [go] req_data["req_path"]: {req_data["req_path"]}')

            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [go] req_data["req_ver"]: {req_data["req_ver"]}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [go] req_data["req_ver"]: {req_data["req_ver"]}')

            
            result = predict(
                req_data["req_path"],
                req_data["req_ver"]
            )
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [go] result {result}')
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [go] result {result}')
            
            return JSONResponse({"result_status": 200, "result_data": result})
            
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return JSONResponse({"result_status": 500, "result_data": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host=f'{os.getenv("VIP_IP")}', port=int(os.getenv("VIP_PORT")))