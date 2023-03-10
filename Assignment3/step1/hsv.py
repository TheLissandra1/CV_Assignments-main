import cv2
import os
import numpy as np


def channel_extract(im, root, fname, rows, cols, channel):
    image_temp = np.zeros((rows, cols))

    for i in range(0, rows):
        for j in range(0, cols):
            image_temp[i, j] = im[i, j, channel]

    image_temp = image_temp.astype(np.uint8)

    extract_path = os.path.join(root, "channel" + str(channel) + "_" + fname)
    cv2.imwrite(extract_path, image_temp)

    return image_temp


# get hsv image and h, s, v channel
def image_extract(root):
    fnames = os.listdir(root)
    print(fnames)
    for fname in fnames:
        if fname.endswith("png") or fname.endswith("jpg"):
            full_name = os.path.join(root, fname)
            print(fname)
            # show hsv image
            im = cv2.imread(full_name)
            cv2.imshow("image_RGB", im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # get size of image
            rows, cols, _ = im.shape

            im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            # cv2.imshow("image_HSV", im_hsv)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            extract_root = os.path.join(root, "hsv")
            extract_path = os.path.join(extract_root, "hsv_" + fname)
            cv2.imwrite(extract_path, im_hsv)

            # H channel
            hsv_0 = channel_extract(im_hsv, extract_root, fname, rows, cols, channel=0)
            # cv2.imshow("HSV_channel_0", hsv_0)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()

            # S channel
            hsv_1 = channel_extract(im_hsv, extract_root, fname, rows, cols, channel=1)
            # cv2.imshow("HSV_channel_1", hsv_1)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()

            # V channel
            hsv_2 = channel_extract(im_hsv, extract_root, fname, rows, cols, channel=2)
            # cv2.imshow("HSV_channel_2", hsv_2)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()


if __name__ == '__main__':
    image_extract("..\Assignment3\step1\Diff\cam1")
    image_extract("..\Assignment3\step1\Diff\cam2")
    image_extract("..\Assignment3\step1\Diff\cam3")
    image_extract("..\Assignment3\step1\Diff\cam4")
