

# detect bad img
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt




def pick_max_contour(contours):
	max_points = 1
	max_index = 0
	max_contour = contours[max_index]
	for i in range(len(contours)):
		#print(contours[i].shape)
		if contours[i].shape[0] > max_points:
			max_points = contours[i].shape[0]
			max_index = i
	max_contour = contours[max_index]
	return max_contour



def find_contour(img):
	# 通过形态学滤波算法找到闭合的磨损区域
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

	contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#max_contour = pick_max_contour(contours)
	#return max_contour
	sorted_contours = sorted(contours, key=lambda a: cv2.contourArea(a), reverse=True)
	return sorted_contours


def compute_area(contour):
	return cv2.contourArea(contour)




if __name__ == "__main__":
	path = './img/bad_case'  #'./img/processed_dataset'  #'./img_example'
	#out_path = './img/processed_dataset_out'
	img_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.jpg') or file.endswith('.png')]

	
	for i  in range(len(img_files)):
		file_path = img_files[i]
		out_path = file_path + "out_.jpg"
		img = cv2.imread(file_path)
		contours = find_contour(img)
		#area = compute_area(contour)
		#print("area: ", area)
		num_contour = 3
		for j in range(num_contour):
			cv2.drawContours(img, contours[j], -1, (0, 255, 0), 10)
			area = compute_area(contours[j])
			print("area: ", area)
		cv2.imwrite(out_path, img)
		
		#cv2.imshow("filtered",img)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()



