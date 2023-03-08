import cv2


def computeImgDiff(background_img_path, foreground_img_path, write_path):
    # Load two images
    img1 = cv2.imread(background_img_path)
    img2 = cv2.imread(foreground_img_path)

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the two grayscale images
    diff = cv2.absdiff(gray1, gray2)

    # Display the difference image
    cv2.imshow('Difference Image', diff)
    cv2.imwrite(write_path, diff)
    print(write_path)
    cv2.waitKey(0)

    # Release the windows and memory
    cv2.destroyAllWindows()


def replaceStrs(strlist, str1, str2):
    new_strlist = []
    for s in strlist:
        newStr = s.replace(str1, str2)
        new_strlist.append(newStr)
    print(new_strlist)
    return new_strlist


def compute_AllDiff(path_list):
    computeImgDiff(path_list[1], path_list[5], path_list[0] + '\diff\Diff_HSV.png')
    computeImgDiff(path_list[2], path_list[6], path_list[0] + '\diff\Diff_H.png')
    computeImgDiff(path_list[3], path_list[7], path_list[0] + '\diff\Diff_S.png')
    computeImgDiff(path_list[4], path_list[8], path_list[0] + '\diff\Diff_V.png')


# replace cam1 to cam2, cam3, cam4
root = '..\Assignment3\step1\Diff\cam1'
background_img_path = '..\Assignment3\step1\Diff\cam1\hsv\hsv_cam1_background.png'
background_img_path_H = '..\Assignment3\step1\Diff\cam1\hsv\channel0_cam1_background.png'
background_img_path_S = '..\Assignment3\step1\Diff\cam1\hsv\channel1_cam1_background.png'
background_img_path_V = '..\Assignment3\step1\Diff\cam1\hsv\channel2_cam1_background.png'
foreground_img_path = '..\Assignment3\step1\Diff\cam1\hsv\hsv_cam1_frame.png'
foreground_img_path_H = '..\Assignment3\step1\Diff\cam1\hsv\channel0_cam1_frame.png'
foreground_img_path_S = '..\Assignment3\step1\Diff\cam1\hsv\channel1_cam1_frame.png'
foreground_img_path_V = '..\Assignment3\step1\Diff\cam1\hsv\channel2_cam1_frame.png'

path_list = [root, background_img_path, background_img_path_H, background_img_path_S, background_img_path_V,
             foreground_img_path, foreground_img_path_H, foreground_img_path_S, foreground_img_path_V]

compute_AllDiff(path_list)

path_list = replaceStrs(path_list, 'cam1', 'cam2')
compute_AllDiff(path_list)
path_list = replaceStrs(path_list, 'cam2', 'cam3')
compute_AllDiff(path_list)
path_list = replaceStrs(path_list, 'cam3', 'cam4')
compute_AllDiff(path_list)

