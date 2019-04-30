from __future__ import print_function
import numpy as np
import argparse
import imutils
import glob
import cv2
import os

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
MIN_MATCH_COUNT = 50

def alignImages(img1, img2,filename,_type):
    print("hi")
    # Initiate SIFT detector
    #   sift = cv2.SIFT()
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img1.shape[:-1]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        matchesMask = None
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params) 
    if _type=="cropped":
        cv2.imwrite(os.path.join('./registration_sift/',filename), img3)
    else:
        cv2.imwrite(os.path.join('./uncropped_sift/',filename), img3)

print("Enter the required options : ")
print("1) Automatic Registration on all images")
print("2) Enter name of visible and thermal image yourself")
option = int(input())
if(option==1):
    directory = 'thermal'
    thermal_images_files=[]
    for filename in os.listdir(directory):
        thermal_images_files.append(filename)
    directory = 'visible'
    visible_images_files=[]
    for filename in os.listdir(directory):
        visible_images_files.append(filename)


    for i in range(len(thermal_images_files)):
        try:
            filename='./results_sift/'+str(i)+'.jpg'
            if os.path.exists(filename):
                print("image already exists")
                continue
            else:
                print("performing registration")
                # load the image image, convert it to grayscale, and detect edges
                template = cv2.imread('thermal/'+thermal_images_files[i])
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                template = cv2.Canny(template, 50, 200)
                (tH, tW) = template.shape[:2]
            
                # loop over the images to find the template in

                # load the image, convert it to grayscale, and initialize the
                # bookkeeping variable to keep track of the matched region
                image = cv2.imread('visible/'+visible_images_files[i])
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                found = None

                # loop over the scales of the image
                for scale in np.linspace(0.2, 1.0, 20)[::-1]:
                    # resize the image according to the scale, and keep track
                    # of the ratio of the resizing
                    resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
                    r = gray.shape[1] / float(resized.shape[1])

                    # if the resized image is smaller than the template, then break
                    # from the loop
                    if resized.shape[0] < tH or resized.shape[1] < tW:
                        break

                    # detect edges in the resized, grayscale image and apply template
                    # matching to find the template in the image
                    edged = cv2.Canny(resized, 50, 200)
                    result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
                    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

                    # if we have found a new maximum correlation value, then update
                    # the bookkeeping variable
                    if found is None or maxVal > found[0]:
                        found = (maxVal, maxLoc, r)

                # unpack the bookkeeping variable and compute the (x, y) coordinates
                # of the bounding box based on the resized ratio
                (_, maxLoc, r) = found
                (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
                (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

                # draw a bounding box around the detected result and display the image
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
                crop_img = image[startY:endY, startX:endX]

                name = "thermal/"+thermal_images_files[i]
                thermal_image = cv2.imread(name, cv2.IMREAD_COLOR)

                crop_img = cv2.resize(crop_img, (thermal_image.shape[1], thermal_image.shape[0]))

                cv2.imwrite(os.path.join('./output_sift/', str(i)+'.jpg'),crop_img)

                final = np.concatenate((crop_img, thermal_image), axis = 1)
                cv2.imwrite(os.path.join('./results_sift/', str(i)+'.jpg'),final)

                cv2.waitKey(0)

                # Read reference image
                refFilename = "thermal/"+thermal_images_files[i]
                print("Reading reference image : ", refFilename)
                imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

                # Read image to be aligned
                imFilename = "output_sift/"+str(i)+'.jpg'
                print("Reading image to align : ", imFilename);  
                im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
                file_name=thermal_images_files[i]
                alignImages(im,imReference,file_name,"cropped")
                alignImages(image,imReference,file_name,"uncropped")

        except:
            pass

else:
    print("Enter the name of thermal image in thermal folder")
    thermal = raw_input()
    print("Enter the name of visible image in visible folder")
    visible = raw_input()
    image = cv2.imread('visible/'+visible+'.jpg')
    template = cv2.imread('thermal/'+thermal+'.jpg')

    print("performing registration")
    # load the image image, convert it to grayscale, and detect edges
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]

    # loop over the images to find the template in

    # load the image, convert it to grayscale, and initialize the
    # bookkeeping variable to keep track of the matched region
    image = cv2.imread('visible/'+visible+'.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None

    # loop over the scales of the image
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])

        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

    # draw a bounding box around the detected result and display the image
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    crop_img = image[startY:endY, startX:endX]

    name = "thermal/"+thermal+'.jpg'
    thermal_image = cv2.imread(name, cv2.IMREAD_COLOR)

    crop_img = cv2.resize(crop_img, (thermal_image.shape[1], thermal_image.shape[0]))

    cv2.imwrite(os.path.join('./output_sift/', thermal+'.jpg'),crop_img)

    final = np.concatenate((crop_img, thermal_image), axis = 1)
    cv2.imwrite(os.path.join('./results_sift/', thermal+'.jpg'),final)

    cv2.waitKey(0)

    # Read reference image
    refFilename = "thermal/"+thermal+'.jpg'
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = "output_sift/"+thermal+'.jpg'
    print("Reading image to align : ", imFilename);  
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
    file_name=thermal+'.jpg'
    alignImages(im,imReference,file_name,"cropped")
    alignImages(image,imReference,file_name,"uncropped")