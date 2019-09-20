from __future__ import division
import time
from util import *
from detection import detection
from detection2 import detection2
import gc


def detect(loaded_ims, imlist, loaded_ori_ims):
    t = time.time()
    imlist, loaded_ims, output = det1.detect(loaded_ims, imlist)
    print("第一阶段检测完毕,用时：%fs，总用时：%fs" % (time.time() - t, time.time() - start))

    """
    list(map(lambda x: det1.write(x, loaded_ims), output))
    
    det_names = det2.det_name(imlist, '_det1')
    
    list(map(cv2.imwrite, det_names, loaded_ims))
    """

    print('\n***** 进入细检阶段: *****')
    # 细检

    locals = det2.locals_info(output)

    t = time.time()
    dashboard = det2.dashboards_img(loaded_ori_ims, locals, imlist)
    print("获取表盘完毕,用时：%fs，总用时：%fs" % (time.time() - t, time.time() - start))

    t = time.time()
    imlist2, loaded_ims2, output2 = det2.detect(dashboard, imlist)
    print("第二阶段检测完毕,用时：%fs，总用时：%fs" % (time.time() - t, time.time() - start))
    # print(getrefcount(dashboard))
    del(dashboard)
    gc.collect()
    output2 = output2.cpu().numpy().astype(float)

    t = time.time()
    digits = det2.digits(output2, imlist2)
    digits_ori = det2.digits_ori(loaded_ori_ims, det2.merge_locals(imlist, locals), digits)
    print("获取所有检测结果，并还原到大图上,用时：%fs，总用时：%fs" % (time.time() - t, time.time() - start))

    t = time.time()
    list(map(lambda x: det2.write(x, loaded_ori_ims), digits_ori))
    det_names2 = det2.det_name(imlist, '_det2')
    list(map(cv2.imwrite, det_names2, loaded_ori_ims))
    print("可视化,用时：%fs，总用时：%fs" % (time.time() - t, time.time() - start))

    print("\n***** 数据处理： *****")
    t = time.time()
    det2.save_anno(imlist, digits_ori, loaded_ori_ims)
    print("保存检测结果到 .txt 文件,用时：%fs，总用时：%fs" % (time.time() - t, time.time() - start))

    t = time.time()
    det2.save_yolo(imlist, digits_ori, loaded_ori_ims)
    print("保存 YOLO 格式的标签到 .txt 文件,用时：%fs，总用时：%fs" % (time.time() - t, time.time() - start))

    t = time.time()
    det2.save_digit(imlist, digits_ori)
    print("只保存读数到 .txt 文件,用时：%fs，总用时：%fs" % (time.time() - t, time.time() - start))

if __name__ == '__main__':
    det1 = detection()
    det2 = detection2()

    start = time.time()

    loaded_ims, imlist = det1.prepare()
    # loaded_ori_ims = copy.deepcopy(loaded_ims)  # 深拷贝
    loaded_ori_ims = loaded_ims

    print("装载权重和图像,用时：%fs" % (time.time() - start))

    detect(loaded_ims, imlist, loaded_ori_ims)

