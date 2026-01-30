import cv2 as cv
import numpy as np

main_img = cv.imread('photo.jpg', cv.IMREAD_REDUCED_COLOR_2)

obj_img = cv.imread('object.png', cv.IMREAD_REDUCED_COLOR_2)

result = cv.matchTemplate(main_img, obj_img, cv.TM_CCOEFF_NORMED)


min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

print('Best match top left: %s' % str(max_loc))
print("Confidence: %s" % max_val)


obj_w = obj_img.shape[1]
obj_h = obj_img.shape[0]


top_left = max_loc
bottom_right = (top_left[0] + obj_w, top_left[1] + obj_h)

cv.rectangle(main_img, top_left, bottom_right, color=(0,255,0), thickness=2, lineType=cv.LINE_4)
cv.imshow('Result', main_img)
cv.waitKey()