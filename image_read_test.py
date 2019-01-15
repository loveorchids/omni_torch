import cv2, time, os, glob
import scipy.misc as scm
from PIL import Image

def test_cv2(path):
    start_time = time.time()
    for pic in glob.glob(path + "/*.jpg"):
        image = cv2.imread(pic)
        image = cv2.resize(image, (320, 320))
    print("--- %s seconds(OpenCV) ---" % (time.time() - start_time))
    
def test_cv2_rgb(path):
    start_time = time.time()
    for pic in glob.glob(path + "/*.jpg"):
        image = cv2.cvtColor(cv2.imread(pic), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 320))
    print("--- %s seconds(OpenCV_convert_to_RGB) ---" % (time.time() - start_time))
    
def test_PIL_grey(path):
    start_time = time.time()
    for pic in glob.glob(path + "/*.jpg"):
        image = Image.open(pic).convert("I")
        image = image.resize((320, 320))
    print("--- %s seconds(PIL grey) ---" % (time.time() - start_time))
    
def test_PIL_color(path):
    start_time = time.time()
    for pic in glob.glob(path + "/*.jpg"):
        image = Image.open(pic).convert("RGB")
        image = image.resize((320, 320))
    print("--- %s seconds(PIL RGB) ---" % (time.time() - start_time))
    
def test_scm(path):
    start_time = time.time()
    for pic in glob.glob(path + "/*.jpg"):
        image = scm.imread(pic)
        image = scm.imresize(image, (320, 320))
    print("--- %s seconds(SCM) ---" % (time.time() - start_time))

if __name__ == "__main__":
    path = os.path.expanduser("~/Pictures/dataset/buddha/trainA")
    test_cv2(path)
    test_cv2_rgb(path)
    test_PIL_grey(path)
    test_PIL_color(path)
    test_scm(path)