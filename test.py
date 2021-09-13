import cv2
import numpy as np
# from matplotlib import pyplot as plt
def grammar_correction(img, grammar=0.4):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, 0.4) * 255.0, 0, 255)
    res = cv2.LUT(img, lookUpTable)
    return res

def convertScale(img, alpha, beta):
    """Add bias and gain to an image with saturation arithmetics. Unlike
    cv2.convertScaleAbs, it does not take an absolute value, which would lead to
    nonsensical results (e.g., a pixel at 44 with alpha = 3 and beta = -210
    becomes 78 with OpenCV, when in fact it should become 0).
    """

    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)

# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=25):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = convertScale(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)


image = cv2.imread('./A/03.JPG')
auto_result, alpha, beta = automatic_brightness_and_contrast(image)

print('alpha', alpha)
print('beta', beta)
#cv2.imshow('auto_result', auto_result)
cv2.imwrite('auto_result.png', auto_result)
#cv2.imshow('image', image)


#--------------------------------------------------------------------
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import numpy as np
import cv2 
import matplotlib.pyplot as plt
image1 = grammar_correction(cv2 .imread('./A/03.JPG', cv2.IMREAD_COLOR))       # queryImage
image2 = grammar_correction(cv2 .imread('./A/04.JPG', cv2.IMREAD_COLOR)) # trainImage

# compute difference
difference = cv2.subtract(image2, image1)

# color the mask red
Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
difference[mask != 255] = [255,255,255]
difference[mask == 255] = [0,0,0]

d = np.zeros_like(difference, np.uint8)
d[mask != 255] = image1[mask != 255]
cv2.imwrite('diff3.png', d)
tmp = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
_,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
b, g, r = cv2.split(d)
rgba = [b,g,r, alpha]
diffTransparent = cv2.merge(rgba,4)
# for i,pixel in enumerate(diffTransparent):
#     print(diffTransparent[i])


# add the red mask to the images to make the differences obvious
image2[mask != 255] = [0,0,255]
image1[mask != 255] = [0, 0, 255]



# store images
cv2.imwrite('diffOverImage1.png', image1)
cv2.imwrite('diffOverImage2.png', image2)
cv2.imwrite('diff.png', difference)
cv2.imwrite('mask2.png', mask)

#-----------------------------------------------------------------------------
# make diff using SSIM
from skimage.metrics import structural_similarity
import cv2
import numpy as np

before = grammar_correction(cv2.imread('./A/03.JPG', cv2.IMREAD_COLOR))
after = grammar_correction(cv2.imread('./A/04.JPG', cv2.IMREAD_COLOR))

# Convert images to grayscale
before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# Compute SSIM between two images
(score, diff) = structural_similarity(before_gray, after_gray, full=True)
print("Image similarity", score)

# The diff image contains the actual image differences between the two images
# and is represented as a floating point data type in the range [0,1] 
# so we must convert the array to 8-bit unsigned integers in the range
# [0,255] before we can use it with OpenCV
diff = (diff * 255).astype("uint8")

# Threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

mask = np.zeros(before.shape, dtype='uint8')
filled_after = after.copy()

for c in contours:
    area = cv2.contourArea(c)
    if area > 40:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.drawContours(mask, [c], 0, (0,255,0), -1)
        cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

cv2.imwrite('before.png', before)
cv2.imwrite('after.png', after)
cv2.imwrite('diff2.png',diff)
cv2.imwrite('mask.png',mask)
cv2.imwrite('filled after.png',filled_after)


#-----------------------------------------------------------------------------
# FLANN matching
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('./A/03.JPG',cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread('./A/04.JPG',cv.IMREAD_GRAYSCALE) # trainImage
# Initiate SIFT detector
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.75*n.distance:
        matchesMask[i]=[1,0]

      
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.imshow(img3,),plt.show()

#--------------------------------------------------------------------
# Alignment
import cv2
import numpy as np

MAX_FEATURES = 20000
GOOD_MATCH_PERCENT = 0.01

def alignImages(im1, im2):

    
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2, None)

    i = cv2.drawKeypoints(im1, keypoints1, None, color=[0,255,0])
    plt.imshow(i), plt.show()
    # Match features.
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2, None)
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)


    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:10]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches2.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h



# Read reference image
refFilename = "./A/03.JPG"
print("Reading reference image : ", refFilename)
imReference = grammar_correction(cv2.imread(refFilename, cv2.IMREAD_COLOR))

# Read image to be aligned
imFilename = "./A/04.JPG"
print("Reading image to align : ", imFilename);
im = grammar_correction(cv2.imread(imFilename, cv2.IMREAD_COLOR))

print("Aligning images ...")
# Registered image will be resotred in imReg.
# The estimated homography will be stored in h.
imReg, h = alignImages(im, imReference)

# Write aligned image to disk.
outFilename = "aligned.jpg"
print("Saving aligned image : ", outFilename);
cv2.imwrite(outFilename, imReg)

# Print estimated homography
print("Estimated homography : \n",  h)

#-------------------------------------------------------------------------------
import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = grammar_correction(cv2.imread("./A/03.JPG", cv2.IMREAD_COLOR))
img2 = grammar_correction(cv2.imread("./A/04.JPG", cv2.IMREAD_COLOR))
# Convert images to grayscale
im1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
im2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initiate SIFT detector
orb = cv2.ORB_create(10000)

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(im1Gray,None)
kp2, des2 = orb.detectAndCompute(im2Gray,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

des1 = np.float32(des1)
des2 = np.float32(des2)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
cv2.imwrite("orbFlann.png", img3)
plt.imshow(img3,),plt.show()
#---------------------------------------------------------------------
