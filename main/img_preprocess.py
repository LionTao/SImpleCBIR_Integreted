import cv2


def preprocess_a(img):
    # Histogram equalization ====
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(ycrcb)

    cv2.equalizeHist(channels[0], channels[0])

    ycrcb = cv2.merge(channels)
    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    # ===========================

    # Canny =====================
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(ycrcb)

    blur = cv2.GaussianBlur(channels[0], (5, 5), 0)
    edge = cv2.Canny(blur, 20, 200)
    channels[0] = (channels[0].astype('float64') - edge * 10).astype('uint8')

    ycrcb = cv2.merge(channels)
    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    # ===========================

    return img


def preprocess_b(img):
    # Histogram equalization ====
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(ycrcb)

    cv2.equalizeHist(channels[0], channels[0])

    ycrcb = cv2.merge(channels)
    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    # ===========================

    # Gaussian ==================
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # ===========================

    return img


if __name__ == '__main__':
    raw_img = cv2.imread('lena.png', 1)
    # cv2.imshow("raw", raw_img)
    # preprocess(raw_img)
