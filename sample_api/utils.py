import cv2
import numpy as np

def center_img(img, bg, mask):
    (h1, w1) = img.shape[:2]
    (h2, w2) = bg.shape[:2]
    
    height = h2
    r = height / float(h1)
    width = int(w1 * r)
    
    if width <= w2:
        dim = (int(w1 * r), height)
    else:
        width = w2
        r = width / float(w1)
        dim = (width, int(h1 * r))

    resized = cv2.resize(img, dim)
    mask = cv2.resize(mask, dim)
    
    # print("<><>", img.shape)
    # print("<><>", bg.shape)
    # print("<><>", resized.shape)
    
    h, w = resized.shape[:2]
    hh, ww = bg.shape[:2]

    yoff = round((hh-h)/2)
    xoff = round((ww-w)/2)

    temp1 = bg[yoff:yoff+h, xoff:xoff+w].copy()
    temp1 = np.bitwise_and(temp1, np.invert(mask))
    
    temp2 = np.bitwise_and(resized, mask)
    temp3 = np.bitwise_or(temp1, temp2)
    
    bg[yoff:yoff+h, xoff:xoff+w] = temp3
    
    return bg