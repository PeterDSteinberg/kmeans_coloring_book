# -*- coding: utf-8 -*-
""" 

I (Peter Steinberg) started this as a demonstration of kmeans,
mainly a demonstration for myself to see how a kmeans fit to images 
would be influenced by diversity of colors and their proportions of 
the image.  

It turns out to also make good coloring book pages for my son.  In that
usage, the best pictures to run through the script are ones with strong 
contrasting features, like an airplane in sky.

Let me know what you think...

The idea is to do this:

	Put some images in ./raw_images 
	Then:

	ipython -i kmeans_coloring_book.py 
  
	    Prompts you to select a photo to make a coloring book page out of.

	    When it pops up, draw a triangle for the training region of kmeans.

	    Change the number of colors (kmeans classes) with:

		    colors 5


"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.colors as mc
from matplotlib.collections import PatchCollection
from scipy import misc
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
import sys
import time
import traceback
from pylab import get_current_fig_manager
center_pcent = 100.0
SAMP_SIZE_MAX = 3000
median_filter = None
IMAGE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__),'raw_images'))
files = os.listdir(IMAGE_DIR)
file_choices = "\n".join(["%d:  %s"%(idx + 1, f) for idx, f in enumerate(files)])
n_colors = 6

def make_drawing(img_name, 
				n_colors = 3, 
				median_filter= median_filter, 
				center_pcent=center_pcent,
				debug=False):
	selected_corners = []
	# Load the Summer Palace photo
	image_object = misc.imread(img_name)
	# Convert to floats instead of the default 8 bits integer coding. Dividing by
	# 255 is important so that plt.imshow behaves works well on float data (need to
	# be in the range [0-1]
	image_object = image_object / 255.0
	answer = "y"
	selected_corners_all = []
	fig = plt.figure(1)
	ax1 = fig.add_axes([0, 0, 1, 1], frameon=False)
	plt.imshow(image_object)
	plt.axis('off')
	plt.show(block=False)
	selected_corners = []
	selected_corners.append(plt.ginput(n=3))
	x = [_[0] for _ in selected_corners[-1]]
	y = [_[1] for _ in selected_corners[-1]]
	ax1.plot(x,
			 y,
			'bo', 
			fillstyle="full",
			markersize=15)
	
	pol = ax1.fill(x, y, fill=False, hatch='//')
	plt.draw()
	plt.show(block=False)
	print('''

Step 1.  Click 3 points of a triangle.
Step 2.  Press red button when done with triangle.
Step 3.  Wait for pictures....
''')
	if debug:
		print('Selected verticies of triangle',selected_corners)
		print('Ok.  Next step...')
	w, h, d = original_shape = tuple(image_object.shape)
	wm, hm = int(w/2.0), int(h/2.0)
	cp = center_pcent / 100.0 / 2.0
	wr, hr = int(w * cp), int(h * cp)
	lowx = wm - wr if  wm - wr > 0 else 0
	hix = wm + wr if wm + wr < w else w
	lowy = hm - hr if  hm - hr > 0 else 0
	hiy = hm + hr if  hm + hr < h else h
	if debug:
		print('cp,wr,hr,lowx,lowy,hix,hiy',cp,wr,hr,lowx,lowy,hix,hiy)
	image_object_center = image_object[lowy:hiy,lowx:hix,:]
	assert d == 3,'Sorry that image will not work'
	image_array = np.reshape(image_object_center, (image_object_center.shape[0] *image_object_center.shape[1], d))

	print("Fitting model")
	t0 = time.time()
	image_array_sample = shuffle(image_array, random_state=0)
	kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample[:SAMP_SIZE_MAX])
	print("Fit it in %0.2f s" % (time.time() - t0))

	# Get labels for all points
	print("Predicting color indices on the full image (k-means)")
	t0 = time.time()
	labels = kmeans.predict(image_object.reshape((w *h, d)))
	if debug:
		print('labels = \n\n\n\n%r'%labels.shape)
		print('w,h,d,wr,hr,wm,hm',w,h,d,wr,hr,wm,hm)
		print("done in %0.3f seconds." % (time.time() - t0))
		print("done in %0.3f seconds." % (time.time() - t0))
		plt.figure(2)
		plt.clf()
		ax = plt.axes([0, 0, 1, 1])
		plt.axis('off')
		plt.title('Subset of original image useful for math.')
		plt.imshow(image_object_center)

		redone1 = predict_from(kmeans.cluster_centers_, labels, wr, hr, as_colors=True)
		plt.figure(3)
		plt.clf()
		ax = plt.axes([0, 0, 1, 1])
		plt.axis('off')
		plt.title('Quantized image (%d colors, K-Means)'%n_colors)
		plt.imshow(redone1)
	hatch_choices_0 =  [ "//" , r"\\" , ",," , "--" , "++" , "xx" , "ooo"  , ".." , "**"]
	hatch_choices= []
	idx = 0
	while len(hatch_choices) < n_colors:
		if idx == 0:
			hatch_choices.extend(hatch_choices_0)
		else:	
			hatch_choices.extend((h1+h2 for h1 in hatch_choices for h2 in hatch_choices_0))
	fig = plt.figure(4)
	redone = predict_from(kmeans.cluster_centers_, 
										labels, 
										w, 
										h, 
										median_filter = median_filter)
	
	xx = range(0, h)
	yy = range(w - 1, -1, -1)
	lk = len(kmeans.cluster_centers_)
	my_cm = mc.from_levels_and_colors(range(lk), [[0.0, 0.0, 0.0,0.5]]*(lk+1), 
									extend = "both")[0]
	cs = plt.contour(xx, 
					yy, 
					redone, 
					n_colors, 
					vmin = 0,
					vmax = n_colors,
					#hatches=hatch_choices,
                  cmap= my_cm,
                  extend='both', 
                  alpha=0.5)
	plt.show(block=False)

def predict_from(codebook, labels, w, h, as_colors = False, median_filter = None,debug = False):
	"""Recreate the (compressed) image from the code book & labels"""
	if as_colors:
		d = codebook.shape[1]
		image = np.zeros((w, h, d))
	else:
		image = np.zeros(( w, h))
	label_idx = 0
	for i in range(w):
		for j in range(h):
			lab = labels[label_idx]
			if as_colors:
				image[i][j] = codebook[lab]
			else:
				image[i][j] = lab
			label_idx += 1
	if median_filter:
		return np.round(ndimage.median_filter(image, size= median_filter))
	return image
def main(n_colors = n_colors, 
		median_filter = median_filter, 
		center_pcent = center_pcent, 
		debug = False):
	help = lambda: sys.stdout.write('Choose a number for the picture you would like:\n\n\n'+file_choices+'\n')
	help()
	global which_file
	try_again = lambda: sys.stdout.write("Try again.  \n")
	which_number_func = lambda: raw_input('Which Number ? [enter]  or [colors %s]'%n_colors).strip().lower()
	while True:
		in_options = False
		try:
			which_number = which_number_func()
			cparts = which_number.split()
			if len(cparts) == 2 and cparts[0].startswith('c'):
				try:
					n_colors = int(cparts[1])
					print('Now using %s colors'%n_colors)
					continue
				except:
					try_again()
					continue
			try:
				which_file = os.path.join(IMAGE_DIR, files[int(which_number.strip())  - 1])
			except: 
				continue
		except KeyboardInterrupt:
			raise 
		except Exception as e:
			print('Skipped %s, with tb=%s'%(repr(e), traceback.format_exc()))
			try_again()
		help()
		make_drawing(which_file, 
					n_colors=n_colors, 
					median_filter = median_filter,
					center_pcent = center_pcent)
		answer = raw_input('Do another ? [Y=yes, N = no]').strip().lower()
		if answer.startswith('y'):
			plt.close('all')
			help()
		else:
			break

if __name__ == "__main__":
	
	n_colors = int(sys.argv[1]) if len(sys.argv) > 1 else n_colors
	median_filter = int(sys.argv[2]) if len(sys.argv) > 2 else median_filter
	center_pcent = float(sys.argv[3]) if len(sys.argv) > 3 else center_pcent
	debug = '-d' in sys.argv
	if center_pcent > 100:
		center_pcent = 100.0
	def redo():
		main(n_colors = n_colors, 
			median_filter = median_filter, 
			center_pcent = center_pcent, 
			debug = debug)
	redo()

