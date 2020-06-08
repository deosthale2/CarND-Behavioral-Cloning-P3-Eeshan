import numpy as np
import cv2
import glob

def copy_images(src, dest):
	dest_filenames = glob.glob(dest + '*.jpg')
	src_filenames = glob.glob(src+'*.jpg')
	for filename in src_filenames:
		img = cv2.imread(filename)
		img_name = filename.split(src)[-1]
		cv2.imwrite(dest+img_name, img)

if __name__ == '__main__':
	src = './data_eeshan/IMG/'
	dest = './data/IMG/'
	copy_images(src, dest)