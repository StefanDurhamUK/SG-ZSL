import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as scio
# import plot_utils
import utils
import torch
import torch.utils.data as dt
# from sklearn.manifold import TSNE
from torch.autograd import Variable
from transformers import DistilBertTokenizer, DistilBertModel

from openTSNE import TSNE
from args import Args
args = Args().parse()

class ProData:
	def __init__(self, args):
		self.args = args
		self.device = self.args.cuda_device
		self.dataset = args.dataset
		self.data_path = args.dataRoot
		self.dataset_path = os.path.join(self.data_path, self.dataset, 'res101.mat')
		self.att_split_path = os.path.join(self.data_path, self.dataset, 'att_splits.mat')
		self.att_unseen_path = os.path.join(self.data_path, self.dataset, 'att_unseen.mat')
		self.label_unseen_path = os.path.join(self.data_path, self.dataset, 'label_unseen.mat')
		self.att_seen_path = os.path.join(self.data_path, self.dataset, 'att_seen.mat')
		self.label_seen_path = os.path.join(self.data_path, self.dataset, 'label_seen.mat')
		self.classes_txt_path = os.path.join(self.data_path, self.dataset, 'trainvalclasses.txt')
		self.vf_path = os.path.join(self.data_path, self.dataset, 'trainvalclasses_WE')
		self.COLOUR = ['DC143C', '800080', '0000FF', '00FFFF', '3CB371', 'FFD700', 'FF8C00', '8B0000', '000000',
		               'FF00FF', 'D2691E', '808080']
	
	# Obtain unique class-id during test period
	def obtain_classes(self):
		features, labels, index = self.obtain_original_data()
		train_val_loc = torch.LongTensor(index['trainval_loc'].astype(np.int32).squeeze() - 1)
		test_unseen_loc = torch.LongTensor(index['test_unseen_loc'].astype(np.int32).squeeze() - 1)
		seen_classes = torch.unique(torch.index_select(labels, 0, train_val_loc).int())
		unseen_classes = torch.unique(torch.index_select(labels, 0, test_unseen_loc).int())
		return seen_classes, unseen_classes
	
	# Read data from mat file, return features, labels, index(for splitting train/test/val dataset)
	def obtain_original_data(self):
		ori_data = scio.loadmat(self.dataset_path)
		index = scio.loadmat(self.att_split_path)
		features = torch.tensor(ori_data['features'].T, dtype=torch.float)
		labels = torch.Tensor(ori_data['labels'].astype(int) - 1)
		return features, labels, index
	
	# Split data for training, test, valuation
	def split_data_by_indices(self, **indices):
		features, labels, index = self.obtain_original_data()
		assert features.shape[0] == labels.shape[0]
		train_idx = torch.LongTensor(index[str(list(indices.values())[0])].astype(np.int32).squeeze() - 1)
		train_features = torch.index_select(features, 0, train_idx)
		train_labels = torch.index_select(labels, 0, train_idx).int()
		if len(indices) == 1:
			return train_features, train_labels
		if len(indices) >= 2:
			val_idx = torch.LongTensor(index[str(list(indices.values())[1])].astype(np.int32).squeeze() - 1)
			val_features = torch.index_select(features, 0, val_idx)
			val_labels = torch.index_select(labels, 0, val_idx)
			if len(indices) == 2:
				return train_features, train_labels, val_features, val_labels
			if len(indices) == 3:
				test_idx = torch.LongTensor(index[str(list(indices.values())[2])].astype(np.int32).squeeze() - 1)
				test_features = torch.index_select(features, 0, test_idx)
				test_labels = torch.index_select(labels, 0, test_idx)
				return train_features, train_labels, val_features, val_labels, test_features, test_labels
			if len(indices) > 3:
				print('parameter errors')
	
	# Split train/val dataset with certain proportion
	def split_data_by_proportion(self, total_samples_num):
		indices = list(range(total_samples_num))
		np.random.shuffle(indices)
		train_indices, val_indices = indices[int(total_samples_num * self.args.val_split):], indices[:int(
			total_samples_num * self.args.val_split)]
		return train_indices, val_indices
	
	# Labels realignment
	@staticmethod
	def realign_labels(old_labels):
		if type(old_labels) is np.ndarray:
			unique_old_labels = np.unique(old_labels)
			new_labels = np.ones_like(old_labels)
		else:
			unique_old_labels = torch.unique(old_labels)
			new_labels = torch.ones_like(old_labels)
		for i in range(unique_old_labels.shape[0]):
			for j in range(old_labels.shape[0]):
				if old_labels[j] == unique_old_labels[i]:
					new_labels[j] = i
		
		return new_labels
	
	# Create Dataloader
	def create_dataloader(self, features, labels, batch_size, drop_last=True, need_realignment=False):
		if need_realignment:
			labels = self.realign_labels(labels)
		dataset = dt.TensorDataset(features, labels)
		dataloader = dt.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
		return dataloader
	
	# Convert data format from cpu to gpu
	def cuda_data(self, *para):
		new_data = [v.to(self.args.cuda_device) for v in list(para)[0]]
		return new_data
	
	@staticmethod
	def decompose_dataset(dataloader, output_type='numpy'):
		if output_type == 'numpy':
			image_batches, label_batches = zip(*[batch for batch in dataloader])
			features, labels = np.array(
				[j.to('cpu').numpy() for i in image_batches for j in i]), np.array(
				[j.to('cpu').numpy() for i in label_batches for j in i]).squeeze()
			return features, labels
		elif output_type == 'tensor':
			pass
	
	# This function aim to read classes name from various dataset and store those class name into csv file
	def store_cls_name_into_csv(self, txt_path):
		if not os.path.exists(os.path.join(txt_path)):
			print("It not existed {} file, please double check it.".format(txt_path))
		else:
			file_name = txt_path.split('/')[-1].split('.')[0]
			with open(txt_path, 'r') as f:
				# read txt
				lines = [line.strip().replace('+', ' ') for line in f.readlines()]
				labels = [[idx, classes] for idx, classes in enumerate(lines, start=0)]
				# store id+labels in csv file
				name = ['labels_id', 'labels']
				store_info = pd.DataFrame(columns=name, data=labels)
				save_dir = os.path.join(self.data_path, self.dataset)
				store_info.to_csv(os.path.join(save_dir, file_name + '.csv'), index=False)
		print('You have stored the "{}.csv" file in {}'.format(file_name, save_dir))
	
	# Generate word-embedding by using BERT
	def bert_semantic_information(self, labels_csv_path):
		if not os.path.exists(os.path.join(labels_csv_path)):
			print("It not existed {} file, please double check it.".format(labels_csv_path))
		else:
			csv_file_name = labels_csv_path.split('/')[-1].split('.')[0]
			path_to_save_wb = csv_file_name + '_WE'
			if not os.path.exists(os.path.join(self.data_path, self.dataset, path_to_save_wb)):
				os.mkdir(os.path.join(self.data_path, self.dataset, path_to_save_wb))
			path_to_save_features = os.path.join(self.data_path, self.dataset, path_to_save_wb)
			data = pd.read_csv(labels_csv_path)
			class_id = data.loc[:, ['labels_id']].values.squeeze()
			class_name = data.loc[:, ['labels']].values.squeeze()
			tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
			model = DistilBertModel.from_pretrained('distilbert-base-uncased')
			model.eval()
			model.cuda()
			device = torch.device("cuda")
			assert len(class_id) == len(class_name)
			f = 0
			for i in range(len(class_id)):
				label = class_id[i]
				feat = ()
				if f >= 0:
					for sent in class_name:
						inputs = tokenizer.encode_plus(
							sent,
							add_special_tokens=True,
							return_tensors='pt',
						)
						input_ids2 = torch.tensor(tokenizer.encode(sent)).unsqueeze(0)  # Batch size 1
						input_ids2 = input_ids2.to(device)
						with torch.no_grad():
							outputs2 = model(input_ids2)
							o2 = outputs2[0].to('cpu').numpy()
						feat += tuple(o2)
					scio.savemat(path_to_save_features + '/' + str(label) + '_' + str(class_name[i]) + '.mat',
					             {'feat_v': feat, 'GT': label})
				f += 1
			print('Finished generate word-embedding from "{}.csv", '
			      'stored in the "{}"'.format(csv_file_name, path_to_save_features))
	
	# Create samples for generator(for examples, bert information of 50 classes become 20000 bert information)
	def create_repetitive_samples_for_generator(self, semantic_file_path, semantic_info_categories, num_of_repetitions,
	                                            task_categories, att_classes_number):
		features = []
		labels = []
		
		def repeat_samples(unique_features, unique_labels, number_of_repetitions, device):
			rep_features = torch.Tensor(
				np.array([j for j in unique_features for i in range(number_of_repetitions)])).to(device)
			rep_labels = torch.Tensor(
				np.array([j for j in unique_labels for i in range(number_of_repetitions)])).int().to(device)
			return rep_features, rep_labels
		
		def load_att(att_path, label_path, att, label):
			a_data = scio.loadmat(att_path)
			a_features = a_data[att].T
			labels_path = scio.loadmat(label_path)
			a_labels = labels_path[label].squeeze()
			return a_features, a_labels
		
		if semantic_info_categories == "bert":
			sem_files = [i for i in os.listdir(semantic_file_path) if i.endswith('.mat')]
			for f in sem_files:
				visual_data = scio.loadmat(os.path.join(semantic_file_path, f))
				cls_label = visual_data["GT"].squeeze()
				cls_feature = visual_data["feat_v"][0][cls_label][0]
				features.append(cls_feature)
				labels.append(cls_label)
			sem_features, sem_labels = repeat_samples(features, labels, num_of_repetitions, self.args.cuda_device)
			return sem_features, sem_labels
		
		elif semantic_info_categories == "att":
			if task_categories == 'gzsl':
				data_a = scio.loadmat(self.att_split_path)
				features = data_a['att'].T
				labels = np.arange(0, att_classes_number, 1)
			elif task_categories == 'zsl_seen':
				features, labels = load_att(self.att_seen_path, self.label_seen_path, 'att_seen', 'label_seen')
			elif task_categories == 'zsl_unseen':
				features, labels = load_att(self.att_unseen_path, self.label_unseen_path, 'att_unseen', 'label_unseen')
			else:
				print('error parameters for task categories!')
			sem_features, sem_labels = repeat_samples(features, labels, num_of_repetitions, self.args.cuda_device)
			return sem_features, sem_labels
		
		elif semantic_info_categories == "wv":
			if task_categories == 'gzsl':
				data_a = scio.loadmat(self.att_split_path)
				features = data_a['wv'].T
				labels = np.arange(0, att_classes_number, 1)
			elif task_categories == 'zsl_seen':
				features, labels = load_att(self.att_seen_path, self.label_seen_path, 'att_seen', 'label_seen')
			elif task_categories == 'zsl_unseen':
				features, labels = load_att(self.att_unseen_path, self.label_unseen_path, 'att_unseen', 'label_unseen')
			else:
				print('error parameters for task categories!')
			sem_features, sem_labels = repeat_samples(features, labels, num_of_repetitions, self.args.cuda_device)
			return sem_features, sem_labels
		
		elif semantic_info_categories == "pure_noises":
			sem_files = [i for i in os.listdir(semantic_file_path) if i.endswith('.mat')]
			for f in sem_files:
				visual_data = scio.loadmat(os.path.join(semantic_file_path, f))
				cls_label = visual_data["GT"].squeeze()
				cls_feature = visual_data["feat_v"][0][cls_label][0]
				features.append(cls_feature)
				labels.append(cls_label)
			sem_features, sem_labels = repeat_samples(features, labels, num_of_repetitions, self.args.cuda_device)
			sem_features = torch.zeros([1, self.args.noiseLen], dtype=torch.float32)
			for i in range(sem_labels.size(0)):
				pure_noise = torch.randn(1, self.args.noiseLen).to(args.cuda_device)
				sem_features = pure_noise if i == 0 else torch.cat((sem_features, pure_noise), 0)
			sem_features.requires_grad = True
			return sem_features, sem_labels
	
	def scale_data_by_cls_and_num(self, features, labels, sel_cls_num, per_cls_samples_num, random_selected=True,
	                              marked_labels=None):
		unique_labels = np.unique(labels)
		assert len(unique_labels) >= sel_cls_num, 'There are not enough {} classes '.format(sel_cls_num)
		if random_selected and unique_labels is not None:
			marked_labels = np.random.choice(unique_labels, sel_cls_num, replace=False)
		selected_features = np.empty(shape=[0, self.args.resSize])
		selected_labels = np.empty(shape=[0])
		for i in marked_labels:
			marked_total_index = np.argwhere(labels == i).squeeze()
			assert len(
				marked_total_index) >= per_cls_samples_num, \
				'There are not enough {} data when label = {}, and just only {}'.format(
					per_cls_samples_num, i, len(marked_total_index))
			marked_selected_index = np.random.choice(marked_total_index, per_cls_samples_num, replace=False)
			selected_features = np.append(selected_features, features[marked_selected_index], axis=0)
			selected_labels = np.append(selected_labels, labels[marked_selected_index], axis=0)
		if random_selected:
			return selected_features, selected_labels, marked_labels
		else:
			return selected_features, selected_labels
	
	# Save result for drawing epoch-acc line graph
	def store_res_for_line_graph(self, result, model_name):
		lg_saved_folder = os.path.join(self.args.line_graph_filepath, self.args.dataset, self.args.framework,
		                               self.args.task_categories, self.args.cur_time)
		if not os.path.exists(lg_saved_folder):
			os.makedirs(lg_saved_folder)
		file_path = os.path.join(lg_saved_folder,
		                         '{}_{}.xlsx'.format(model_name, self.args.need_quality_check))
		output = open(file_path, 'w', encoding='gbk')
		for i in range(len(result)):
			for j in range(len(result[i])):
				output.write(str(result[i][j]))
				output.write('\t')
			output.write('\n')
		output.close()
	
	# Draw T-SNE
	def draw_t_sne(self, data_t, data_g, model, marked_labels):
		def obtain_fake_data():
			fake_features = torch.zeros(1, self.args.resSize).to(self.device)
			origin_labels = torch.zeros(1, ).to(self.device)
			with torch.no_grad():
				for i, (images, labels) in enumerate(data_g):
					features = model(Variable(images))
					start_num = 1 if fake_features.size(0) == 1 else 0
					labels = labels.unsqueeze(1)
					if fake_features.size(0) == 1:
						fake_features = features[0].view(1, -1)
						origin_labels = labels[0]
					for j in range(start_num, images.size(0)):
						fake_features = torch.cat((fake_features, features[j].view(1, -1)), 0)
						origin_labels = torch.cat((origin_labels, labels[j]), 0)
			return fake_features, origin_labels
		
		def draw_chart(x, y, save_img_name=None, colors=None, selected_color=True):
			if colors is None:
				colors = self.COLOUR if selected_color else self.random_color(self.args.outSizeTS)
			color_rgb = np.array(list(map(self.hex_to_rgb, colors)))
			for i in range(len(y)):
				plt.scatter(x[i, 0], x[i, 1], c=np.array([color_rgb[int(y[i])] / 255]), label=int(y[i]))
			plt.xlim(-50, 50)
			plt.ylim(-50, 50)
			graph_save_path = os.path.join(self.args.graph_filepath, '{}.png'.format(save_img_name))
			plt.savefig(graph_save_path)
			plt.close()
			
			return colors
		
		generator_features, generator_labels = obtain_fake_data()
		teacher_features, teacher_labels = data_t
		# teacher_labels = self.realign_labels(teacher_labels)
		generator_features, generator_labels = generator_features.to('cpu').numpy(), generator_labels.to('cpu').numpy()
		sel_generator_features, sel_generator_labels = self.scale_data_by_cls_and_num(generator_features,
		                                                                              generator_labels,
		                                                                              self.args.tsne_cls_num,
		                                                                              self.args.cls_samples_num,
		                                                                              random_selected=False,
		                                                                              marked_labels=marked_labels)
		# sel_generator_labels = self.realign_labels(sel_generator_labels)
		sel_generator_labels = sel_generator_labels.reshape((-1, 1)).squeeze(1)
		combine_features = np.vstack((teacher_features, sel_generator_features))
		
		tsne = TSNE(perplexity=self.args.perplexity, exaggeration=self.args.exaggeration, metric="euclidean",
		            n_jobs=2, n_iter=self.args.tsne_n_iter,
		            random_state=42)
		combine_features_2d = tsne.fit(combine_features)
		teacher_features_2d, generator_features_2d = combine_features_2d[:teacher_features.shape[0]], \
		                                             combine_features_2d[teacher_features.shape[0]:]
		
		files_name_suffix = '{}_{}_{}_{}.png'.format(self.args.cur_time, self.args.framework,
		                                             self.args.task_categories, self.args.dataset)
		graph_save_folder = os.path.join(self.args.tsne_graph_filepath, self.args.dataset, self.args.framework,
		                                 self.args.task_categories, self.args.cur_time)
		
		if not os.path.exists(graph_save_folder):
			os.makedirs(graph_save_folder)
		# real features TSNE!
		plt.figure()
		# utils.plot(teacher_features_2d, teacher_labels.astype(np.int64), draw_centers=True, colors=utils.MOUSE_10X_COLORS)
		utils.plot(teacher_features_2d, teacher_labels.astype(np.int64), draw_centers=True)
		ff_save_path = os.path.join(graph_save_folder,
		                            '{}_{}_{}_{}'.format('RF', self.args.perplexity, self.args.exaggeration,
		                                                 files_name_suffix))
		plt.savefig(ff_save_path)
		plt.show()
		plt.close()
		
		# fake features TSNE!
		plt.figure()
		# utils.plot(generator_features_2d, sel_generator_labels.astype(np.int64), draw_centers=True, colors=utils.MOUSE_10X_COLORS)
		utils.plot(generator_features_2d, sel_generator_labels.astype(np.int64), draw_centers=True)
		rf_save_path = os.path.join(graph_save_folder,
		                            '{}_{}_{}_{}'.format('FF', self.args.perplexity, self.args.exaggeration,
		                                                 files_name_suffix))
		plt.savefig(rf_save_path)
		plt.show()
		plt.close()
		# exit(0)
		"""
		combine_features = np.vstack((teacher_features, sel_generator_features))
		t_sne = TSNE(n_components=2, random_state=0, n_iter=self.args.tsne_n_iter, learning_rate=self.args.tsne_lr)
		features_2d = t_sne.fit_transform(combine_features)
		teacher_features_2d, generator_features_2d = features_2d[:teacher_features.shape[0]], features_2d[
																							  teacher_features.shape[
																								  0]:]
		print('begin draw tsne')
		colors = draw_chart(teacher_features_2d, teacher_labels,
							save_img_name='Real_{}_{}_{}_{}_{}_{}_{}'.format(self.args.framework,
																			 self.args.task_categories,
																			 self.args.semantic_type,
																			 self.args.noise_type,
																			 self.args.need_quality_check,
																			 self.args.dataset,
																			 self.args.cur_time))
		print('teacher_tsne finished!')
		colors = draw_chart(generator_features_2d, sel_generator_labels,
							save_img_name='Fake_{}_{}_{}_{}_{}_{}_{}'.format(self.args.framework,
																			 self.args.task_categories,
																			 self.args.semantic_type,
																			 self.args.noise_type,
																			 self.args.need_quality_check,
																			 self.args.dataset,
																			 self.args.cur_time), colors=colors)

		print('generator_tsne finished!')
		print()
		"""

"""
if __name__ == '__main__':
	prodata = ProData(args)
	
	path = '../data/ImNet2/att_splits.mat'
	data = scio.loadmat(path)
	# path_res = '../data/ImNet2/res101.mat'
	# data_res = scio.loadmat(path_res)
	# seen_labels_loc = data['trainval_loc']-1
	# unseen_labels_loc = data['test_unseen_loc']-1
	# all_cls_labels = data_res['labels']
	# seen_labels = all_cls_labels[seen_labels_loc].reshape((160000,-1))
	# unseen_labels = all_cls_labels[unseen_labels_loc].reshape((54000,-1))
	# unique_sl = np.unique(seen_labels)-1
	# unique_ul = np.unique(unseen_labels)-1
	all_attrs = data['att'].T
	seen_attr = all_attrs[:1000]
	unseen_attr = all_attrs[1000:]
	scio.savemat('../data/ImNet2/att_seen.mat', {'att_seen': seen_attr})
	scio.savemat('../data/ImNet2/att_unseen.mat', {'att_seen': unseen_attr})

	#path1 ='../data/ImNet2/att_seen.mat'
	#path2 = '../data/ImNet2/att_unseen.mat'
	#seen_attrs = scio.loadmat(path1)
	#unseen_attrs = scio.loadmat(path2)
	#path_awa = '../data/AWA1/att_seen.mat'
	#awa1_data = scio.loadmat(path_awa)
"""
	
	
