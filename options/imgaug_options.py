import argparse


class ImgAug:
    def initialize(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--crop_size", type=tuple, default=(1200, 1200),
                                 help="size of input images")
        self.parser.add_argument("--img_size", type=tuple, default=(500, 500),
                                 help="size of input images")
        self.parser.add_argument("--img_channel", type=int, default=3,
                                 help="3 stand for color image while 1 for greyscale and n for multilayers")
        
        self.parser.add_argument("--random_crop", type=bool, default=True,
                                 help="randomly crop an image")
        self.parser.add_argument("--random_flip", type=bool, default=True,
                                 help="randomly flip an image")
        self.parser.add_argument("--do_affine", type=bool, default=True,
                                 help="do random translation on image or not")
        self.parser.add_argument("--random_brightness", type=bool, default=True,
                                 help="adjust brightness on image")
        self.parser.add_argument("--random_noise", type=bool, default=False,
                                 help="apply random gaussian noise on image")
        
        self.parser.add_argument("--translation", type=tuple, default=(0.2, 0.2),
                                 help="a list of two elements indicate translation on x and y axis")
        self.parser.add_argument("--scale", type=tuple, default=(0.8, 1.2),
                                 help="a list of two elements indicate scale on x and y axis")
        self.parser.add_argument("--shear", type=tuple, default=(-5, 5),
                                 help="a list of two elements indicate shear on x and y axis")
        self.parser.add_argument("--rotation", type=tuple, default=(-5, 5),
                                 help="a number indicate rotation of image")
        self.parser.add_argument("--project", type=list, default=[2, 2],
                                 help="a list of two elements indicate projection on x and y axis")
        self.parser.add_argument("--custom", type=list,
                                 help="a list of 8 element represent a custom transform matrix")
        self.parser.add_argument("--imgaug_max_delta", type=float, default=0.2,
                                 help="random up bound of brightness adjustment")
        self.parser.add_argument("--imgaug_mean", type=float, default=0.0,
                                 help="mean of random coefficiency of translation")
        self.parser.add_argument("--imgaug_stddev", type=float, default=0.2,
                                 help="standard deviation of random coefficiency of translation")
        self.parser.add_argument("--imgaug_order", type=list, default=[],
                                 help="the order of combining transformation matrix")
        self.parser.add_argument("--imgaug_crop_bbox", type=bool, default=False,
                                 help="crop the bound box from image or not")
