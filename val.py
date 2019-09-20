import os


def get_file_name(dir_path):
    return [x for x in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, x))]


if __name__ == "__main__":
    numDir = 'data/test/number'  # gt
    detNumDir = 'data/test/detect_number'  # 检测得到的结果所在的文件夹名字

    numTxt = sorted([os.path.join(numDir, x) for x in get_file_name(numDir)])
    detNumTxt = sorted([os.path.join(detNumDir, x) for x in get_file_name(detNumDir)])

    count = 0
    all_count = 0
    print("检测出错的图片名字:")
    for i in range(len(numTxt)):
        f1 = open(numTxt[i], 'r')
        txt1 = f1.readlines()
        all_count += len(txt1)
        f2 = open(detNumTxt[i], 'r')
        txt2 = f2.readlines()

        for digit in txt2:
            if digit == '\n':
                continue
            if digit in txt1:
                count += 1
            else:
                # 打印出错的图片名字
                print(detNumTxt[i][-10:-4]+'.jpg')
    print('检测结果：', count, all_count, count/all_count)


