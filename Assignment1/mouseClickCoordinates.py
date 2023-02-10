import cv2
# [print(i) for i in dir(cv2) if 'EVENT' in i]


# function to display the coordinates of
# the points clicked on the image 
def click_event(event, x, y, flags, params):

    # check left mouse clicks
    if event == cv2.EVENT_FLAG_LBUTTON:

        # display the coordinates on the Shell
        print (x, ' ', y)

        # display the coodinates on the image window
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        cv2.putText(img, str(x) + ',' + str(y), (x,y), font, 1, (144,0,255), 2)
        cv2.imshow('image', img)  

    # # check right mouse clicks
    # if event == cv2.EVENT_FLAG_RBUTTON:

    #     # display the coordinates on the Shell
    #     print (x, ' ', y)

    #     # display the coodinates on the image window
    #     font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        


    #     cv2.imshow('image', img)  

if __name__ == "__main__":
    
    # read the image
    fname = "Assignment1\Checkerboards\C1.png"
    img = cv2.imread(fname)
    print(img)
    cv2.namedWindow('image', 0)
    cv2.resizeWindow('image', 1000, 1000)
    # display the image
    cv2.imshow("image", img)
    
    # set mouse handler and call click_event()
    cv2.setMouseCallback("image", click_event)

    cv2.waitKey(0)
    cv2.destroyAllWindows()