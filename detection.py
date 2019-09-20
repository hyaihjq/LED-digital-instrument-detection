from __future__ import division
from util import *
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import random


class detection:
    """ YOLO v3 Detection """
    def __init__(self):
        self.doc = 'detection'
        # Image / Directory containing images to perform detection upon
        self.images = 'data/led/imgs'
        # Image / Directory to store detections to
        self.det = 'data/led/imgs-det'  # 'save'
        # Batch size
        self.bs = 1
        # Object Confidence to filter predictions
        self.confidence = 0.5
        # NMS Threshhold
        self.nms_thresh = 0.4
        # Config file
        self.cfgfile = 'cfg/yolov3-voc.cfg'
        # weights file
        self.weights = 'weights/yolov3-voc_final.weights'  # 'yolov3.weights'
        # Input resolution of the network. Increase to increase accuracy. Decrease to increase speed
        self.reso = 416
        self.num_classes = 12
        self.classes_cfg = 'cfg/voc.names'
        self.classes = load_classes(self.classes_cfg)
        self.colors = pkl.load(open("pallete", "rb"))

    def prepare(self, loaded_ims=False, imlist=False):
        if not imlist:
            # Detection phase
            try:
                imlist = [osp.join(osp.realpath('.'), self.images, img) for img in os.listdir(self.images)]
            except NotADirectoryError:
                imlist = []
                imlist.append(osp.join(osp.realpath('.'), self.images))
            except FileNotFoundError:
                print("No file or directory with the name {}".format(self.images))
                exit()
        else:
            print('imlist')
        if not os.path.exists(self.det):
            os.makedirs(self.det)
        if not loaded_ims:
            loaded_ims = [cv2.imread(x) for x in imlist]
        else:
            print('load_ims')
        return loaded_ims, imlist

    def detect(self, loaded_ims, imlist):
        CUDA = torch.cuda.is_available()

        model = Darknet(self.cfgfile)
        model.load_weights(self.weights)

        model.net_info["height"] = self.reso
        inp_dim = int(model.net_info["height"])
        assert inp_dim % 32 == 0
        assert inp_dim > 32

        # If there's a GPU availible, put the model on GPU
        if CUDA:
            model.cuda()

        # Set the model in evaluation mode
        model.eval()

        im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
        im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

        leftover = 0
        if (len(im_dim_list) % self.bs):
            leftover = 1

        if self.bs != 1:
            num_batches = len(imlist) // self.bs + leftover
            im_batches = [torch.cat((im_batches[i * self.bs: min((i + 1) * self.bs,
                        len(im_batches))])) for i in range(num_batches)]

        write = 0

        if CUDA:
            im_dim_list = im_dim_list.cuda()
        for i, batch in enumerate(im_batches):
            # load the image
            if CUDA:
                batch = batch.cuda()
            with torch.no_grad():
                prediction = model(Variable(batch), CUDA)

            prediction = write_results(prediction, self.confidence, self.num_classes, nms_conf=self.nms_thresh)

            #if type(prediction) == int:

            #   for im_num, image in enumerate(imlist[i * self.bs: min((i + 1) * self.bse, len(imlist))]):
            #       im_id = i * self.bs + im_num
            #    continue

            prediction[:, 0] += i * self.bs  # transform the atribute from index in batch to index in imlist

            if not write:  # If we have't initialised output
                output = prediction
                write = 1
            else:
                output = torch.cat((output, prediction))

            #for im_num, image in enumerate(imlist[i * self.bs: min((i + 1) * self.bs, len(imlist))]):
            #    im_id = i * self.bs + im_num
            #    objs = [self.classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            if CUDA:
                torch.cuda.synchronize()
        try:
            output
        except NameError:
            print("No detections were made")
            exit()

        im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())
        scaling_factor = torch.min(int(self.reso) / im_dim_list, 1)[0].view(-1, 1)
        output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2
        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

        torch.cuda.empty_cache()
        return imlist, loaded_ims, output

    def write(self, x, results):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results[int(x[0])]
        cls = int(x[-1])
        color = random.choice(self.colors)
        label = "{0}".format(self.classes[cls])
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
        return img


if __name__ == '__main__':
    a = detection()
    print(a.images)


    pass


