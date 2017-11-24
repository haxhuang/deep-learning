import numpy as np
from genplate import *


def gen(batch_size=32):
    chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
             "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
             "Y", "Z"
             ];

    M_strIdx = dict(zip(chars, range(len(chars))))
    l_plateStr, l_plateImg = G.genBatch(batch_size, 2, range(31, 65), "./plate_test", (272, 72))
    b = cv2.resize(l_plateImg[0], (272, 72))
    print(b.shape)
    X = np.array(l_plateImg, dtype=np.uint8)
    print(X.shape)
    # print(len(l_plateStr))
    ytmp = np.array(list(map(lambda x: [M_strIdx[a] for a in list(x)], l_plateStr)), dtype=np.uint8)
    print(ytmp[0])
    # print(ytmp.shape)  # (32,7)
    y = np.zeros([ytmp.shape[1], batch_size, len(chars)])  # (7,32,65)
    print(y.shape)

    for batch in range(batch_size):
        for idx, row_i in enumerate(ytmp[batch]):
            y[idx, batch, row_i] = 1
    print(type(y))
    a = np.array([yy for yy in y])
    print("shape:", X.shape, a.shape)


if __name__ == '__main__':
    gen()
