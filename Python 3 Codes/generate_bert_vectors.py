import os
from bert_embedding import BertEmbedding
from os.path import join, isfile
import re
import numpy as np
import pickle
import argparse
import skipthoughts
import h5py

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--caption_file', type=str, default='Data/sample_captions.txt',
					   help='caption file')
	parser.add_argument('--data_dir', type=str, default='Data',
					   help='Data Directory')

	args = parser.parse_args()
	with open( args.caption_file ) as f:
		captions = f.read().split('\n')

	captions = [cap for cap in captions if len(cap) > 0]
	print(captions)

	bert_embedding = BertEmbedding()
	embed_list = []
	embed_sum = np.zeros(768)
	embedding = bert_embedding(captions,'avg')
	for sent in range(len(captions)):
		word_embed_list = embedding[sent][1]
		for word_embed in word_embed_list:
			embed_sum += word_embed
		embed_list.append(embed_sum/len(word_embed_list))
	embed_list_np = np.asarray(embed_list)
	print(embed_list_np)
	print(embed_list_np.shape)

	if os.path.isfile(join(args.data_dir, 'submission_bert_caption_vectors.hdf5')):
		os.remove(join(args.data_dir, 'submission_bert_caption_vectors.hdf5'))
	h = h5py.File(join(args.data_dir, 'submission_bert_caption_vectors.hdf5'))
	h.create_dataset('vectors', data=embed_list_np)		
	h.close()

if __name__ == '__main__':
	main()