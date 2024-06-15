import cv2
import numpy as np
from skimage.filters import threshold_otsu 




def filter_image(image, mask):
    r = image[:,:,0] * mask
    g = image[:,:,1] * mask
    b = image[:,:,2] * mask
    return np.dstack([r,g,b])


def compute_area(image):
	# count zero pixels
	return


def print_piexel(img):
	h, w = img.shape
	print("h: ", h)
	print("w: ", w)
	for i in range(h):
		for j in range(w):
			print(img[i][j])

	return


if __name__ == "__main__":
	print("hello opencv")
	img = cv2.imread("./img/img1.png")
	'''
	gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	res,thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
	edges = cv2.dilate(cv2.Canny(thresh,0,255),None)

    #轮廓检测
    #轮廓检测与排序
	cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
	mask = np.zeros((256,256), np.uint8)
	masked = cv2.drawContours(mask, [cnt],-1, 255, -1)

	dst = cv2.bitwise_and(img, img, mask=mask) #区域分割
	segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
	'''
	
	img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	img_gray=cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)

	thresh = threshold_otsu(img_gray) #找阈值
	print("gray size: ",img_gray.shape)
	print("thresh size: ", thresh)
	img_otsu = img_gray < thresh
	print_piexel(img_otsu)

	filtered = filter_image(img, img_otsu)

	print("filtered shape: ", filtered.shape)
	cv2.imshow("filtered",filtered)
	#cv2.imshow("Image",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()