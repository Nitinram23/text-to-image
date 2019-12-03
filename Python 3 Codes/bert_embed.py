from bert_embedding import BertEmbedding
import numpy as np
import pickle
import argparse
import json
import os
from os.path import join, isfile
import re
import h5py

def save_caption_vectors_flowers(data_dir):
	import time
	
	img_dir = join(data_dir, 'flowers/jpg')
	image_files = [f for f in os.listdir(img_dir) if 'jpg' in f]
	# print(image_files[300:400])
	# print(len(image_files))
	image_captions = { img_file : [] for img_file in image_files }

	caption_dir = join(data_dir, 'flowers/text_c10')
	class_dirs = []
	for i in range(1, 103):
		class_dir_name = 'class_%.5d'%(i)
		class_dirs.append( join(caption_dir, class_dir_name))

	for class_dir in class_dirs:
		caption_files = [f for f in os.listdir(class_dir) if 'txt' in f]
		for cap_file in caption_files:
			with open(join(class_dir,cap_file)) as f:
				captions = f.read().split('\n')
			img_file = cap_file[0:11] + ".jpg"
			# 5 captions per image
			image_captions[img_file] += [cap for cap in captions if len(cap) > 0][0:5]

	encoded_captions = {}
	bert_embedding = BertEmbedding()
	
	for i, img in enumerate(image_captions):
		st = time.time()
		embed_list = []
		embed_sum = np.zeros(768)
		embedding = bert_embedding(image_captions[img],'avg')
		for sent in range(len(image_captions[img])):
			word_embed_list = embedding[sent][1]
			for word_embed in word_embed_list:
				embed_sum += word_embed
			embed_list.append(embed_sum/len(word_embed_list))
		embed_list_np = np.asarray(embed_list)
		encoded_captions[img] = embed_list_np
		print(i, len(image_captions), img)
		print("Seconds", time.time() - st)
		
	h = h5py.File(join(data_dir, 'flower_bert.hdf5'))
	for key in encoded_captions:
		h.create_dataset(key, data=encoded_captions[key])
	h.close()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--split', type=str, default='train',
                       help='train/val')
	parser.add_argument('--data_dir', type=str, default='Data',
                       help='Data directory')
	parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch Size')
	parser.add_argument('--data_set', type=str, default='flowers',
                       help='Data Set : Flowers, MS-COCO')
	args = parser.parse_args()
	
	if args.data_set == 'flowers':
		save_caption_vectors_flowers(args.data_dir)
	else:
		print('incorrect data')

if __name__ == '__main__':
	main()

