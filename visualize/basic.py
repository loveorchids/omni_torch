import os, math
import cv2
import numpy as np

def vis_image(args, tensorlist, epoch, batch, loss, idx=None, concat_axis=1):
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
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = math.sqrt(img.size) / 2500
        font_position = (img.shape[1] - int(130 * font_size * 2), img.shape[0] - int(20 * font_size * 2))
        thickness = round(font_size * 2)
        cv2.putText(img, "loss: "+ str(loss)[:6], font_position, font, font_size, (0, 0, 255), thickness)
        cv2.imwrite(path, img)