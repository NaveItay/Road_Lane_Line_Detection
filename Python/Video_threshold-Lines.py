import cv2
import numpy as np

cap = cv2.VideoCapture('VID_20200329_083617.mp4')

ret, current_frame = cap.read()

# Saves every save_count frame
save_count = 1

# Start names from '0.jpg'
image_name = 10

xmin = 600
xmax = 1750
ymin = 440
ymax = 780
xmiddle = (xmin + xmax) / 2

def triangle(RGB):
    height, width, = MyShape
    triangle = np.array([
                       [(500, 780), (1040, 440), (1750, 780)]
                       ])
    mask = np.zeros((height, width), dtype=np.uint8)
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.rectangle(mask, (940, 410), (1180, 480), (0, 0, 0), -1)
    RGB = cv2.bitwise_and(RGB, RGB, mask=mask)
    return RGB

def lines_filter_right(filter_input):
    filter_1_output = filter_input[filter_input[..., 4] > 20]
    if filter_1_output is not None:
        filter_2_output = filter_1_output[filter_1_output[..., 4] < 40]
    if filter_2_output is not None:
        filter_3_output = filter_2_output[filter_2_output[..., 0] > xmiddle]
    if filter_3_output is not None:
        filter_4_output = filter_3_output[filter_3_output[..., 0] < xmax]
    if filter_4_output is not None:
        filter_5_output = filter_4_output[filter_4_output[..., 2] > xmiddle]
    if filter_5_output is not None:
        filtered = filter_5_output[filter_5_output[..., 2] < xmax]
    return filtered

def lines_filter_left(filter_input):
    filter_1_output = filter_input[filter_input[..., 4] < -20]
    if filter_1_output is not None:
        filter_2_output = filter_1_output[filter_1_output[..., 4] > -40]
    if filter_2_output is not None:
        filter_3_output = filter_2_output[filter_2_output[..., 0] < xmiddle]
    if filter_3_output is not None:
        filter_4_output = filter_3_output[filter_3_output[..., 0] > xmin]
    if filter_4_output is not None:
        filter_5_output = filter_4_output[filter_4_output[..., 2] < xmiddle]
    if filter_5_output is not None:
        filtered = filter_5_output[filter_5_output[..., 2] > xmin]
    return filtered

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

# Dashboard mask
if ret is True:
    MyShape = current_frame[:, :, 0].shape
    Dashboard_mask = np.zeros((MyShape[0], MyShape[1]), dtype=np.uint8)
    Dashboard_mask[:, :] = 255
    Dashboard_mask = triangle(Dashboard_mask)

while(cap.isOpened()):

    image = np.copy(current_frame)
    triangle_frame = triangle(current_frame)
    HSV = cv2.cvtColor(triangle_frame, cv2.COLOR_BGR2HSV)

    # White_mask = cv2.inRange(HSV, lower[White], upper[White])
    Yellow_mask = cv2.inRange(HSV, lower[Yellow], upper[Yellow])
    Orange_mask = cv2.inRange(HSV, lower[Orange], upper[Orange])

    # WhiteResult = cv2.bitwise_and(current_frame, current_frame, mask=White_mask)
    YellowResult = cv2.bitwise_and(current_frame, current_frame, mask=Yellow_mask)
    OrangeResult = cv2.bitwise_and(triangle_frame, triangle_frame, mask=Orange_mask)

    Sum = cv2.bitwise_or(YellowResult, OrangeResult)

    # White_edges = cv2.Canny(WhiteResult, 1, 255)
    # Yellow_edges = cv2.Canny(YellowResult, 1, 255)
    # Orange_edges = cv2.Canny(OrangeResult, 1, 255)
    Sum_edges = cv2.Canny(Sum, 1, 255)

    # Hough Transformation
    Sum_linesP = cv2.HoughLinesP(Sum_edges, 1, np.pi / 180, 10, np.array([]), 60, 150)

    # Draw my lines
    if Sum_linesP is not None:

        Angle_Sum = np.array([(180 / np.pi) * np.arctan2(Sum_linesP[:, :, 3] - Sum_linesP[:, :, 1], Sum_linesP[:, :, 2] - Sum_linesP[:, :, 0])])
        Angle_Sum = Angle_Sum.reshape(Angle_Sum.shape[1], Angle_Sum.shape[0], Angle_Sum.shape[2])
        Sum_data = np.append(Sum_linesP, Angle_Sum, axis=2)

        # RIGHT SIDE FILTER
        right_filtered_lines = lines_filter_right(Sum_data)

        if len(right_filtered_lines) > 1:
            X1_R = np.mean(right_filtered_lines[..., 0])
            X2_R = np.mean(right_filtered_lines[..., 2])
            Y1_R = np.mean(right_filtered_lines[..., 1])
            Y2_R = np.mean(right_filtered_lines[..., 3])

            cv2.line(image, (int(X2_R), int(Y2_R)), (int(X1_R), int(Y1_R)), (255, 0, 0), 8, cv2.LINE_AA)

        # LEFT SIDE FILTER
        left_filtered_lines = lines_filter_left(Sum_data)

        if len(left_filtered_lines) > 1:
            X1_L = np.mean(left_filtered_lines[..., 0])
            X2_L = np.mean(left_filtered_lines[..., 2])
            Y1_L = np.mean(left_filtered_lines[..., 1])
            Y2_L = np.mean(left_filtered_lines[..., 3])

            cv2.line(image, (int(X2_L), int(Y2_L)), (int(X1_L), int(Y1_L)), (255, 0, 0), 8, cv2.LINE_AA)

    showResult = weighted_img(image, current_frame)

    cv2.imshow('sourceImg', showResult)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        print("Save frame " + str(image_name) + ".jpg")
        # Save frame to image file
        cv2.imwrite(str(image_name) + '.jpg', showResult)
        # Next image name
        image_name += 1
        #
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, current_frame = cap.read()

cap.release()
cv2.destroyAllWindows()
