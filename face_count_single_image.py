from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
import matplotlib.pyplot as plt
import PIL.Image, PIL.ImageTk
import tkinter as tk
import tkinter.font as font
from tkinter import filedialog
import time



def cvtToRGB(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


#paths
trained_model='E:\DO AN TIN HOC\Pytorch_Retinaface-master\mobilenet025_resnet50_pretrained\Resnet50_Final.pth'
save_folder='' #save putText
dataset_folder='E:/DO AN TIN HOC/Pytorch_Retinaface-master/test_images/'
save_pic_folder='E:\DO AN TIN HOC\Pytorch_Retinaface-master\results'
#enable things
origin_size=True
cpu=False
save_image=True
#variables
network='resnet50'
vis_thres=0.5
confidence_threshold=0.02
top_k=5000
nms_threshold=0.4
keep_top_k=750
#img_name=''

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    #print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model
######################################GUI#######################################
def count_button():
    global img, photo
    source = img_name
    img = cv2.imread(source)
    count = 0
    gray = cvtToRGB(img)
    a=getsq(gray)
    for (x,y,w,h) in a:
        drawreg(img,x,y,w,h)
        count=count+1
    print('Number of faces: ',count, end =' ')
    im_shape = img.shape
    cv2.putText(img,str(count),(5,im_shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,0),2,cv2.LINE_AA)

    #image and stuff
    height, width, no_channels = img.shape
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))

    #canvas
    canvas = tk.Canvas(frame_2, width = width, height = height)
    canvas.place(x = 0, y = 0)
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)

def browse():
    global img_name
    window.filename = filedialog.askopenfilename(initialdir = "D:/",
    title = "Select a File",
    filetypes = (("all files","*.*"),("jpg files","*.jpg"),("png files","*.png")))
    img_name = window.filename
    global img, photo
    source=img_name
    img = cv2.imread(source)
    if img.shape[0] > 720:
        baseheight = 720
        hpercent = baseheight/ (img.shape[0])
        wsize = int(img.shape[1] * hpercent)
        img  = cv2.resize(img , (wsize,baseheight))
    if img.shape[1] > 1280:
        basewidth = 1280
        wpercent = basewidth/ (img.shape[1])
        hsize = int(img.shape[0]*wpercent)
        img = cv2.resize(img , (basewidth,hsize))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #image and stuff
    height, width, no_channels = img.shape
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))

    #canvas
    # canvas = tk.Canvas(frame_2, width = width, height = height)
    # canvas.place(x = 0, y = 0)
    # canvas.create_image(0, 0, image=photo, anchor=tk.NW)
    image = tk.Label(frame_2, image=photo, bg = 'PeachPuff2')
    image.place(x =0, y = 0)
def save():
    global img, photo, img_save
    save_it = filedialog.asksaveasfile(mode = 'w',
    defaultextension=".jpg",)
    cv2.imwrite(save_it.name, img_save)

def count_button():
    global img, photo, img_name
    source = img_name
    img = cv2.imread(source)
    img_raw = cvtToRGB(img)
    img = np.float32(img_raw)

    # testing scale
    target_size = 1600
    max_size = 2150
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    resize = float(target_size) / float(im_size_min)

    # prevent bigger axis from being more than max_size:
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)
    if origin_size:
        resize = 1

    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    # _t['forward_pass'].tic()
    loc, conf, landms = net(img)  # forward pass
    # _t['forward_pass'].toc()
    # _t['misc'].tic()
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1]
    # order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]


    dets = np.concatenate((dets, landms), axis=1)
    # _t['misc'].toc()


    # save image
    if save_image:
        count=0
        for b in dets:
            if b[4] < vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            count=count+1
            cv2.putText(img_raw, (str(count)), (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            # cv2.putText(img_raw, (str(count)+' ,' + text), (cx, cy),
            #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            # cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            # cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            # cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            # cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

        cv2.putText(img_raw,str(count),(5,im_shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,0),2,cv2.LINE_AA)


        img_raw_RGB = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    global img_save
    img_save = img_raw_RGB
        #print(count)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    if img_raw_RGB.shape[0] > 720:
        baseheight = 720
        hpercent = baseheight/ (img_raw_RGB.shape[0])
        wsize = int(img_raw_RGB.shape[1] * hpercent)
        img_raw_RGB  = cv2.resize(img_raw_RGB , (wsize,baseheight))
    if img_raw_RGB.shape[1] > 1280:
        basewidth = 1280
        wpercent = basewidth/ (img_raw_RGB.shape[1])
        hsize = int(img_raw_RGB.shape[0]*wpercent)
        img_raw_RGB = cv2.resize(img_raw_RGB , (basewidth,hsize))

    img_raw_RGB = cv2.cvtColor(img_raw_RGB, cv2.COLOR_BGR2RGB)
    height, width, no_channels = img_raw_RGB.shape
    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img_raw_RGB))

    #canvas
    image = tk.Label(frame_2, image=photo, bg = 'PeachPuff2')
    image.place(x =0, y = 0)
################################################################################
if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if network == "mobile0.25":
        cfg = cfg_mnet
    elif network == "resnet50":
        cfg = cfg_re50

    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, trained_model, cpu)
    net.eval()
    cudnn.benchmark = False
    device = torch.device("cpu" if cpu else "cuda")
    net = net.to(device)

    # testing dataset
    testset_folder = dataset_folder
    testset_list = dataset_folder[:-7] + "wider_val.txt"
    # _t = {'forward_pass': Timer(), 'misc': Timer()}

    # testing begin
    k=0    #in line 199


    window = tk.Tk()
    window.title("RetinaFace Face Counter")

    #Font
    myFont = font.Font(family='Lucida Calligraphy', size=15, weight='bold')

    #Second frame
    frame_2 = tk.Frame(window, width=1280, height=720 ,bg = 'PeachPuff2')
    frame_2.pack(side = tk.LEFT,padx=10, pady=10)

    #Third frames
    frame_3 = tk.Frame(window, width=270, height=720)
    frame_3.pack(side = tk.LEFT)


    label_1 = tk.Label(frame_3, text = "Face detection \n -Single Image-")
    label_1['font'] = font.Font(family='Calibri', size=30, weight='bold')
    label_1.place(x = 0, y = 60)


    Button_browse = tk.Button(frame_3, text="Upload Image",bg = 'light sea green',fg = 'snow',
    command = browse,
    activebackground = 'gray',width = 15,
    activeforeground = 'snow')
    Button_browse['font'] = myFont
    Button_browse.place(x = 30, y = 180)

    Button_count = tk.Button(frame_3, text="Start Counting",bg = 'light sea green',fg = 'snow',
    command = count_button,width = 15,
    activebackground = 'gray',
    activeforeground = 'snow')
    Button_count['font'] = myFont
    Button_count.place(x = 30, y = 230)

    Button_save = tk.Button(frame_3, text="Save Image",bg = 'light sea green',fg = 'snow',
    command = save,width = 15,
    activebackground = 'gray',
    activeforeground = 'snow')
    Button_save['font'] = myFont
    Button_save.place(x = 30, y = 280)


    img= cv2.imread("./background/bird.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bird_img =  PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(img))
    image = tk.Label(frame_3, image=bird_img)
    image.place(x=30, y=330)


    window.mainloop()
