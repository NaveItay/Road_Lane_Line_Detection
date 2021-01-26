import cv2
import numpy as np
from matplotlib import pyplot as plt

def triangle(RGB):
    height, width, = Gray.shape
    triangle = np.array([
                       [(500, 780), (1040, 440), (1750, 780)]
                       ])
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.rectangle(mask, (940, 410), (1180, 480), (0, 0, 0), -1)
    RGB = cv2.bitwise_and(RGB, RGB, mask=mask)
    return RGB

def weighted_img(img, initial_img, α=0.5, β=0.5, λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

# **** Color filter ****
White = 0
Yellow = 1
Orange = 2
#                     [{h_min},{h_max},{s_min},{s_max},{v_min},{v_max}]
Color_mask = np.array([[0, 44, 8, 23, 203, 219],            #WhiteColor
                      [11, 17, 0, 54, 208, 233],            #YellowColor
                      [18, 27, 70, 136, 158, 255]])           #OrangeColor

upper = Color_mask[:, 1::2]
lower = Color_mask[:, 0::2]

############################################

figsize = (100, 100)

img = cv2.imread('10.jpg')
Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=figsize)
plt.imshow(RGB, cmap="gray", vmin=0, vmax=255)
plt.show()

# Road mask
# Road_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
# Road_mask[500:780, :] = 255

# Dashboard mask
Dashboard_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
Dashboard_mask[:, :] = 255
Dashboard_mask = triangle(Dashboard_mask)
plt.figure(figsize=figsize)
plt.imshow(Dashboard_mask, cmap="gray", vmin=0, vmax=255)
plt.show()


# RGB = cv2.bitwise_and(RGB, RGB, mask=Road_mask)
RGB = triangle(RGB)
image = np.copy(RGB)
plt.figure(figsize=figsize)
plt.imshow(RGB, cmap="gray", vmin=0, vmax=255)
plt.show()

HSV = cv2.cvtColor(RGB, cv2.COLOR_RGB2HSV)
# plt.figure(figsize=figsize)
# plt.imshow(HSV, cmap="gray", vmin=0, vmax=255)
# plt.show()


# White_mask = cv2.inRange(HSV, lower[White], upper[White])
Yellow_mask = cv2.inRange(HSV, lower[Yellow], upper[Yellow])
Orange_mask = cv2.inRange(HSV, lower[Orange], upper[Orange])

# WhiteResult = cv2.bitwise_and(RGB, RGB, mask=White_mask)
YellowResult = cv2.bitwise_and(RGB, RGB, mask=Yellow_mask)
OrangeResult = cv2.bitwise_and(RGB, RGB, mask=Orange_mask)

# plt.figure(figsize=figsize)
# plt.imshow(WhiteResult, cmap="gray", vmin=0, vmax=255)
# plt.show()
#
plt.figure(figsize=figsize)
plt.imshow(YellowResult, cmap="gray", vmin=0, vmax=255)
plt.show()
#
plt.figure(figsize=figsize)
plt.imshow(OrangeResult, cmap="gray", vmin=0, vmax=255)
plt.show()


# White_edges = cv2.Canny(WhiteResult, 1, 255)
Yellow_edges = cv2.Canny(YellowResult, 1, 255)
Orange_edges = cv2.Canny(OrangeResult, 1, 255)

plt.figure(figsize=figsize)
plt.imshow(Yellow_edges, cmap="gray", vmin=0, vmax=255)
plt.show()

plt.figure(figsize=figsize)
plt.imshow(Orange_edges, cmap="gray", vmin=0, vmax=255)
plt.show()




#                                           Rho, Theta,VotingThreshold,MinLineLength, MaxLineGap
# White_linesP = cv2.HoughLinesP(White_edges, 1, np.pi / 180, 10, np.array([]), 200, 250)
Yellow_linesP = cv2.HoughLinesP(Yellow_edges, 1, np.pi / 180, 10, np.array([]), 40, 250)
Orange_linesP = cv2.HoughLinesP(Orange_edges, 1, np.pi / 180, 10, np.array([]), 40, 250)


# Slope_matrix = (Orange_linesP[:,:,3] - Orange_linesP[:,:,1]) / (Orange_linesP[:,:,2] - Orange_linesP[:,:,0])
# DegreeFreedom = 0.1
# IdlLeftSlop = -0.69
# left_slope = Slope_matrix[(IdlLeftSlop - DegreeFreedom * IdlLeftSlop) < Slope_matrix.any() < (IdlLeftSlop + DegreeFreedom * IdlLeftSlop)]
#
# right_slope = Slope_matrix

# Draw my White lines
# if White_linesP is not None:
#     length = len(White_linesP)
#     for i in range(0, length):
#         x1, y1, x2, y2 = White_linesP[i][0]
#         angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
#         if not(-34 <= angle <= 23) and not(85 <= angle <= 95):
#         # if (25 <= angle <= 70) | (110 <= angle <= 150):
#             cv2.line(RGB, (x1, y1), (x2, y2), (255, 255, 255), 10, cv2.LINE_AA)

# Draw my Yellow lines
if Yellow_linesP is not None:
    length = len(Yellow_linesP)
    for i in range(0, length):
        x1, y1, x2, y2 = Yellow_linesP[i][0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
        if not(-20 <= angle <= 40) and not(85 <= angle <= 95):
        # if (25 <= angle <= 70) | (110 <= angle <= 150):
            cv2.line(RGB, (x1, y1), (x2, y2), (255, 0, 0), 1, cv2.LINE_AA)

# Draw my Orange lines
if Orange_linesP is not None:
    length = len(Orange_linesP)
    for i in range(0, length):
        x1, y1, x2, y2 = Orange_linesP[i][0]
        angle = np.arctan2(y2 - y1, x2 - x1) * (180. / np.pi)
        if not(-20 <= angle <= 40) and not(85 <= angle <= 95):
        # if (25 <= angle <= 70) | (110 <= angle <= 150):
            cv2.line(RGB, (x1, y1), (x2, y2), (255, 0, 0), 1, cv2.LINE_AA)



RGB = weighted_img(RGB, image)

plt.figure(figsize=figsize)
plt.imshow(RGB, cmap="gray", vmin=0, vmax=255)
plt.show()