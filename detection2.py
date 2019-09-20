from __future__ import division
from util import *
import os.path as osp
import random
from detection import detection


class detection2(detection):
    def __init__(self):
        super().__init__()
        self.doc = 'detection2'
        self.reso = 608
        # Number of digits strings
        self.row = 4
        # 数字字符串包含的数字的个数
        self.col = 4
        # 小数点存在的个数
        self.point = 2
        # crop[0]：是否裁截,以及待裁截区域大约以表盘区域为中心,默认高向外延伸 crop[1] 倍，宽延伸 crop[2] 倍
        self.crop = 1, 2, 2.5  # tuple
        # 保存裁截后的图
        self.save_crop = False
        # True:存储所有的检测结果到 .txt
        self.save_all_result = False
        # True:保存检测结果为 YOLO 格式并且存到 .txt
        self.save_labels = False
        # True:只存储最终的读数到 .txt
        self.save_only_digit = False
        # True:测量时钟
        self.clock = True
        # local 的类别编号
        self.local_classID = 10
        # point 的类别编号
        self.point_classID = 11
        # True:显示检测出来的每一个目标
        self.draw_number = False

    def locals_info(self, output):
        """
        从 output 中保留最后一列类别为 10 的行
        """

        locals = output[output[:, -1] == self.local_classID]
        return locals

    def merge_locals(self, imlist, locals):
        """
        把某一张图上多个 local 坐标合并，得到表盘的位置
        :param imlist:
        :param locals: 含有所有类别为 self.local_classID 的信息的列表（即 self.locals_info 的输出）
        :return:合并后的表盘坐标
        """
        local_value = locals[0, -1].item()  # 最后一列即是 local 类别对应的数字
        # locals shape:[tensor,tensor,...]
        locals = [locals[locals[:, 0] == x] for x in range(len(imlist))]
        l_max = [np.array(x).max(axis=0) for x in locals]
        l_min = [np.array(x).min(axis=0) for x in locals]
        merge_locals = []
        [merge_locals.append([i, l_min[i][1], l_min[i][2], l_max[i][3], l_max[i][4], 0, 0, local_value]) for i in
         range(len(imlist))]
        return merge_locals

    def det_name(self, imlist, add_str='', suffix='.jpg'):
        """
        保存的图片名字
        :param imlist:
        :return:
        """
        name = [osp.split(x)[-1] for x in imlist]
        name = [osp.splitext(x)[0] for x in name]
        det_name = [osp.join(self.det, x) + add_str + suffix for x in name]
        return det_name

    def extent_local(self, im, local):  # , im_name):
        """
        将合并之后的 local 扩大以作检测
        :param im:
        :param local: 具体的某一个合并后的表盘信息
        :return:将表盘区域扩大后得到的新的坐标
        """
        h_ori, w_ori = im.shape[:2]
        xmin, ymin, xmax, ymax = [int(x) for x in local[1:5]]
        h = ymax - ymin
        w = xmax - xmin
        ymin -= int(h * self.crop[1])
        ymax += int(h * self.crop[1])
        xmin -= int(w * self.crop[2])
        xmax += int(w * self.crop[2])

        # 有些表盘过于的小，小于 32 像素的边，扩大。
        if xmax - xmin < 32:
            xmin -= 32
            xmax += 32
        if ymax - ymin < 32:
            ymin -= 32
            ymax -= 32

        # 确定裁剪的区域在原图内
        ymin = ymin if ymin > 0 else 1
        ymax = ymax if ymax < h_ori else h_ori - 1
        xmin = xmin if xmin > 0 else 1
        xmax = xmax if xmax < w_ori else w_ori - 1
        extent_local = [xmin, ymin, xmax, ymax]
        return extent_local

    def dashboards_img(self, loaded_ims, locals, imlist):
        def crop_img(im, local, im_name):
            """
            获取表盘的图像，作下一次检测
            :param im: image
            :param local: local position
            :param im_name: image name
            :return: dashboard image
            """
            xmin, ymin, xmax, ymax = self.extent_local(im, local)
            dashboard_img = im[ymin:ymax, xmin:xmax]  # 图像即是矩阵，先对 1 轴切片。先 y 再 x 相当于图片里的先 w 再 h
            # if xmax - xmin < 32 or ymax - ymin < 32:
            #    dashboard_img = cv2.resize(dashboard_img, (64, 64))
            if self.save_crop:
                try:
                    cv2.imwrite(im_name, dashboard_img)

                except:
                    print('{0} {1} {2}'.format('Image', im, 'save failed\n'))
            return dashboard_img

        dashboards_img = list(map(crop_img, loaded_ims, self.merge_locals(imlist, locals), self.det_name(imlist, '_crop')))
        return dashboards_img

    def digits(self, output, imlist):
        digits = []
        def inlocal(number, local, extent=0.5):
            """
            判断检测出来的 number 是否在 local 内部
            :param number:
            :param local:
            :param extent: 是否放大 local 所在的区域，
            放大的策略是只要 number 与 local 有接触，便认为 number 在 local 内部
            :return:
            """
            xmin, ymin, xmax, ymax = [int(x) for x in number[1:5]]
            xcentre = (xmin + xmax) // 2
            ycentre = (ymin + ymax) // 2
            w = xmax - xmin
            h = ymax - ymin
            lxmin, lymin, lxmax, lymax = [int(x) for x in local[1:5]]
            if extent:
                if xcentre >= lxmin - extent * w and xcentre <= lxmax + extent * w and \
                        ycentre >= lymin - extent * h and ycentre <= lymax + extent * h:
                    return True
                else:
                    return False
            else:
                if xcentre >= lxmin and xcentre <= lxmax and ycentre >= lymin and ycentre <= lymax:
                    return True
                else:
                    return False

        for im_ID in range(len(imlist)):
            single_img_info = [x for x in output if x[0] == im_ID]
            locals = [x for x in single_img_info if x[-1] == self.local_classID]
            locals_score = sorted([(x[5] + x[6]) for x in locals], reverse=True)
            try:
                gap = locals_score[self.row]
            except:
                gap = 0
            # 将位置得分和类别得分相加，取前面 self.row 个
            # 如果取的很多，而实际上只检测了较少的个数，那么理应获得所有的检测。
            locals = [x for x in locals if x[5] + x[6] > gap]
            # numbers 包含数字和小数点的坐标信息
            numbers = [x for x in single_img_info if x[-1] != self.local_classID]
            for local in locals:
                # number 只包含数字的位置信息
                number = [x for x in numbers if x[-1] != self.point_classID and inlocal(x, local)]
                number_score = sorted([(x[5] + x[6]) for x in number], reverse=True)
                try:
                    gap = number_score[self.col]
                except:
                    gap = 0
                number = [x for x in number if x[5] + x[6] > gap]

                point = [x for x in numbers if x[-1] == self.point_classID and inlocal(x, local)]
                point_score = sorted([(x[5] + x[6]) for x in point], reverse=True)
                try:
                    gap = point_score[self.point]
                except:
                    gap = 0
                point = [x for x in point if x[5] + x[6] > gap]
                info = number + point
                t = [[x[1] + x[3], x[-1]] for x in info]
                t = sorted(t, key=lambda x: x[0])
                result = ''
                for x in t:
                    x = str(int(x[1])) if x[1] != self.point_classID else '.'
                    result += str(x)
                if self.clock:
                    result = result.replace('..', ':', 1)
                try:
                    result = str(float(result))  # 去掉多余的零
                except:
                    pass
                local = list(local)
                local[5] = result
                digits += [local] + [list(x) for x in info]
        # print(self.doc + ':det2.digits is running')
        return digits

    def digits_ori(self, loaded_ori_ims, locals, digits):
        """
        把第二次检测（在小图上）的坐标，还原到大图上
        :param loaded_ori_ims: 原图
        :param locals: 真实表盘的位置 （ self.merge_locals() 的输出）
        :param digits: 已经得出整个表盘示数字符串的 output
        :return: 输入的 digits 是在小图上的坐标，将其还原到大图上
        """
        digits_ori = []

        def process(im, local, digit):
            bias_x, bias_y = self.extent_local(im, local)[0:2]
            digit[1] += bias_x
            digit[2] += bias_y
            digit[3] += bias_x
            digit[4] += bias_y
            return digit

        for digit in digits:
            i = int(digit[0])
            im = loaded_ori_ims[i]
            local = locals[i]
            digits_ori.append(process(im, local, digit))

        return digits_ori

    def write(self, x, results):
        img = results[int(x[0])]
        cls = int(x[-1])
        color = random.choice(self.colors)

        c1 = tuple([int(x) for x in x[1:3]])
        c2 = tuple([int(x) for x in x[3:5]])
        label = "{0}".format(self.classes[cls]) if cls != self.local_classID else str(x[5])
        if cls == self.local_classID:
            cv2.rectangle(img, c1, c2, [106, 133, 202], 3)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 1)[0]
            c2 = c1[0] + t_size[0]//2 + 3, c1[1] + t_size[1]//2 +1
            # c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4  # default
            cv2.rectangle(img, c1, c2, color, -1)
            # cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [225, 255, 255], 2)  # default
            cv2.putText(img, label, (c1[0], c1[1] + 10), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)  # 2, 2
        else:
            if self.draw_number:
                cv2.rectangle(img, c1, c2, color, 1)
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
                cv2.rectangle(img, c1, c2, color, -1)
                cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img

    def save_anno(self, imlist, digits, loaded_ims):
        if not self.save_all_result:
            return
        txt_names = self.det_name(imlist, '_info', '.txt')
        for i, txt in enumerate(txt_names):
            with open(txt, 'w') as f:
                # 保存的 txt 信息第一行为图片的分辨率
                h, w = loaded_ims[i].shape[:2]
                f.write(str(h) + ' ' + str(w) + '\n')
                for digit in digits:
                    if digit[0] == i:
                        # print(digit[5])
                        t = [str(int(x)) + ' ' for x in digit[1:5]]
                        # 对最后一列 digit[-1] == self.local_classID 的数据而言，digit[5] 即表表盘的读数
                        # 而对最后一列 digit[-1] ！= self.local_classID 的，digit[5] 是 float 型，不能与 str 相加，
                        # 不含有表盘读数，设置为 none
                        try:
                            content = t[0] + t[1] + t[2] + t[3] + str(int(digit[7])) + ' ' + digit[5] + '\n'
                        except:
                            content = t[0] + t[1] + t[2] + t[3] + str(int(digit[7])) + ' none' + '\n'
                        f.write(content)

    def save_yolo(self, imlist, digits, loaded_ims):
        if not self.save_labels:
            return
        txt_names = self.det_name(imlist, '_yolo', '.txt')
        for i, txt in enumerate(txt_names):
            with open(txt, 'w') as f:
                h, w = loaded_ims[i].shape[:2]
                for digit in digits:
                    if digit[0] == i:
                        x1, y1, x2, y2 = [int(x) for x in digit[1:5]]
                        label = int(digit[-1])
                        xc = (x1 + x2) / 2.0 / w
                        yc = (y1 + y2) / 2.0 / h
                        w_ = (x2 - x1) / w
                        h_ = (y2 - y1) / h
                        content = str(label) + ' ' + str(xc) + ' ' + str(yc) + ' ' \
                                 + str(w_) + ' ' + str(h_) + '\n'
                        f.write(content)

    def save_digit(self, imlist, digits):
        if not self.save_only_digit:
            return
        txt_names = self.det_name(imlist, '', '.txt')
        for i, txt in enumerate(txt_names):
            with open(txt, 'w') as f:
                for digit in digits:
                    if digit[0] == i:
                        try:
                            # digit[5] 若为数字,则加上字符'\n'便会出错，
                            # 若不出错说明 digit[5] 为需要的字符串。
                            content = digit[5] + '\n'
                        except:
                            content = ''
                        f.write(content)


if __name__ == "__main__":
    det2 = detection2()

