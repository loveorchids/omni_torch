import os
import cv2
import numpy as np

def vis_image(args, tensorlist, epoch, batch, idx=None, concat_axis=1):
    if type(idx) is int:
        idx = [idx]
    if not idx:
        idx = range(max([t.size(0) for t in tensorlist]))
    for i in idx:
        img = np.concatenate([t[i].data.to("cpu").numpy().squeeze() * 255
                              for t in tensorlist], axis=concat_axis)
        img = cv2.bitwise_not(img.astype("uint8"))
        img_name = str(epoch)+"_"+str(batch)+"_"+str(i)+".jpg"
        path = os.path.join(os.path.expanduser(args.log), img_name)
        cv2.imwrite(path, img)