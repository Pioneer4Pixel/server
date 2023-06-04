import _thread
import argparse
import os
import time
from queue import Queue
from PIL import Image
import filetype
import cv2
import numpy as np
import torch
from torch.nn import functional as F
# import pickle
# from demo import load_checkpoints
from rife.ssim import ssim_matlab
from rife.RIFE_HDv3 import Model


device = "cuda" if torch.cuda.is_available() else "cpu"
animeGanModel = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", device=device).eval()
face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", device=device)
image_format = "jpeg"

# generator, kp_detector = pickle.load(config_path='first-order-model/config/vox-256.yaml', 
#                             checkpoint_path='vox-cpk.pth.tar')

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
count = 0

def load(model_path: str = 'rife/flownet-v46.pkl', fp16: bool = False):
    global model # pylint: disable=global-statement
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if fp16:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Model()
    model.load_model(model_path, -1)
    model.eval()
    model.device()


def interpolate(input_dir: str, output_dir: str, scale: float=1.0, multi: int=30, change: float=0.3, buffer_frames: int=20, fp16: bool=False):
    model_path = os.path.join(os.path.dirname(__file__), 'rife/flownet-v46.pkl')
    if model is None:
        load(model_path, fp16)
    videogen = []
    for f in os.listdir(input_dir):
        print(f)
        fn = os.path.join(input_dir, f)
        if os.path.isfile(fn) and filetype.is_image(fn):
            videogen.append(fn)
    videogen = sorted(videogen)
    print(f'input images: {videogen}')
    # videogen.sort(key=lambda x:int(os.path.basename(x[:-4])))
    frame = cv2.imread(videogen[0], cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
    h, w, _ = frame.shape
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    def write(output_dir, buffer):
        global count # pylint: disable=global-statement
        item = buffer.get()
        while item is not None:
            cv2.imwrite(f'{output_dir}/{count:0>6d}.jpg', item[:, :, ::-1])
            item = buffer.get()
            count += 1

    def execute(I0, I1, n):
        if model.version >= 3.9:
            res = []
            for i in range(n):
                res.append(model.inference(I0, I1, (i+1) * 1. / (n+1), scale))
            return res
        else:
            middle = model.inference(I0, I1, scale)
            if n == 1:
                return [middle]
            first_half = execute(I0, middle, n=n//2)
            second_half = execute(middle, I1, n=n//2)
            if n % 2:
                return [*first_half, middle, *second_half]
            else:
                return [*first_half, *second_half]

    def pad(img):
        return F.pad(img, padding).half() if fp16 else F.pad(img, padding)

    tmp = max(128, int(128 / scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    buffer = Queue(maxsize=8192)
    _thread.start_new_thread(write, (output_dir, buffer))

    print('image', videogen[0], 'ssim', 0.99, f'buffer {buffer_frames} frames')
    for _i in range(buffer_frames): # fill starting frames
        buffer.put(frame)

    I1 = pad(torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
    for f in videogen:
        frame = cv2.imread(f, cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
        I0 = I1
        I1 = pad(torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
        I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
        if ssim > 0.99: # skip duplicate frames
            continue
        if ssim < change:
            output = []
            for _i in range(buffer_frames): # fill frames if change rate is above threshold
                output.append(I0)
            for _i in range(buffer_frames):
                output.append(I1)
        else:
            output = execute(I0, I1, multi-1)
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            buffer.put(mid[:h, :w])
        buffer.put(frame)
        print('image', f, 'ssim', round(ssim.item(), 2), f'buffer {buffer_frames} frames' if ssim < change else f'create {len(output)} frames')

    print('image', videogen[-1], 'ssim', 0.99, f'buffer {buffer_frames} frames')
    for _i in range(buffer_frames): # fill ending frames
        buffer.put(frame)
    while not buffer.empty():
        time.sleep(0.5)

def animate(folderName):
    for frame in os.listdir(f"{folderName}"):
        im_in = Image.open(f"{folderName}/{frame}").convert("RGB")
        im_out = face2paint(animeGanModel, im_in, side_by_side=False)
        im_out.save(f"{folderName}/{frame}")
    
# def cartoon(foldername):
#     for frame in os.listdir(f"{foldername}"):
#         im_in = Image.open(f"{foldername}/{frame}").convert("RGB")
#         im_out = cartoonize(animeGanModel, im_in)
#         im_out.save(f"{foldername}/{frame}")    

