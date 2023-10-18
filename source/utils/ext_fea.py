import threading
from threading import Lock
import torch
import clip
import cv2
import os
import time
from PIL import Image

num_threads = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

done = []
lock = Lock()

directory = 'videos'

def ext_fea():
    with torch.no_grad():
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            if os.path.isfile(f):
                lock.acquire()
                if filename in done:
                    lock.release()
                    continue
                done.append(filename)
                lock.release()
                if(os.path.exists("vid_fea/%s.pt"%filename)):
                    continue
                vidcap = cv2.VideoCapture(f)
                success,image = vidcap.read()
                count = 0
                video_fea = torch.zeros(1,1)
                while success:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_pil = Image.fromarray(image)
                    image_fea = model.encode_image(preprocess(image_pil).unsqueeze(0).to(device))
                    if count == 0:
                        video_fea = image_fea.cpu()
                    else:
                        video_fea = torch.cat([video_fea, image_fea.cpu()])
                    success,image = vidcap.read()
                    count += 1
                torch.save(video_fea, 'vid_fea/%s.pt'%filename)
                
def status_report():
    while True:
        total = len(os.listdir(directory))
        lock.acquire()
        current = len(done)
        lock.release()
        if total != current:
            print('Current progress: %6d/%6d'%(current,total), end = '\r')
            time.sleep(0.5)
        else:
            print('Completed task')
            break
        
logger = threading.Thread(target = status_report, name = 'logger')
logger.start()
        
threads = []
for x in range(num_threads):
    x += 1
    threads.append(threading.Thread(target = ext_fea, name = 't%d'%x))
    
for x in threads:
    x.start()
    
for x in threads:
    x.join()

logger.join()