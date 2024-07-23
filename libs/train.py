import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import re
# from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from opacus import PrivacyEngine
from opacus.utils.module_modification import convert_batchnorm_modules

sys.path.append("/home/fan/code/AZSL_Ray_Tune/libs")

import models as models
from pro_data import ProData
from new_loss import NewLoss, CenterLoss

# sys.path.append("libs")
sys.path.append("checkpoints")


# writer = SummaryWriter('/home/stefan/code/ZSL_TGS/runs/teacher_model', comment='teacher_model')


class Training:
	def __init__(self, args):
		self.args = args
		
		self.device = self.args.cuda_device
		self.pro_data = ProData(self.args)
		self.new_loss = NewLoss()
		# Loss
		self.BCE_loss = nn.BCELoss().to(self.device)
		self.MSE_loss = nn.MSELoss().to(self.device)
		self.CRE_loss = nn.CrossEntropyLoss().to(self.device)
		# File path for word embedding
		self.train_cls_wb = os.path.join(args.dataRoot, args.dataset, 'trainvalclasses_WE')  # 40 classes WE(train+ val)
		self.test_cls_wb = os.path.join(args.dataRoot, args.dataset, 'testclasses_WE')  # 10 classes WE(test_unseen)
		self.all_cls_wb = os.path.join(args.dataRoot, args.dataset, 'allclasses_WE')  # 50 classes WE(all classes)
		# Model
		self.teacher = models.Teacher(self.args.resSize, self.args.hidSizeTSZ_1, self.args.hidSizeTSZ_2,
		                              self.args.outSizeTS, self.args.drop_p)
		self.generator = models.Generator(self.args.in_dim_G, self.args.noiseLen, self.args.hidSizeG, self.args.resSize,
		                                  self.args.drop_p)
		self.student = models.Student(self.args.resSize, self.args.hidSizeTSZ_1, self.args.hidSizeTSZ_2,
		                              self.args.outSizeTS, self.args.drop_p)
		self.z_net = models.Z_net(self.args.resSize, self.args.hidSizeTSZ_1, self.args.hidSizeTSZ_2,
		                          self.args.outSizeZ, self.args.drop_p)
		
		# traditional center loss
		# self.center_loss = CenterLoss(num_classes=self.args.outSizeTS, feat_dim=self.args.resSize)
		# self.g_center_param = list(self.generator.parameters()) + list(self.center_loss.parameters())
		# self.G_Center_opti = optim.Adam(self.g_center_param, lr=self.args.lrG, weight_decay=self.args.g_weight_decay)
		
		# new center loss
		self.center_loss = CenterLoss(feat_dim=self.args.hidSizeTSZ_2)
		# self.g_center_param = list(self.generator.parameters()) + list(self.center_loss.parameters())
		# self.G_Center_opti = optim.Adam(self.center_loss.parameters(), lr=self.args.lrG)
		
		# Optimizer
		self.T_opti = optim.Adam(self.teacher.parameters(), lr=self.args.lrT, weight_decay=self.args.t_weight_decay)
		self.G_opti = optim.Adam(self.generator.parameters(), lr=self.args.lrG, weight_decay=self.args.g_weight_decay)
		self.S_opti = optim.Adam(self.student.parameters(), lr=self.args.lrS, weight_decay=self.args.s_weight_decay)
		self.Z_opti = optim.Adam(self.z_net.parameters(), lr=self.args.lrZ, weight_decay=self.args.z_weight_decay)
		
		# Path for saving and loading model
		save_path = os.path.join(self.args.checkPointRoot, self.args.framework, self.args.task_categories)
		save_path = save_path if self.args.task_categories == 'GZSL_all' else os.path.join(save_path,
		                                                                                   self.args.AZSL_test)
		self.save_path = os.path.join(save_path, self.args.dataset)
		self.best_model_path = os.path.join(save_path, self.args.dataset, 'best_model')
		# lw: new loss weight; nl: noise length; sn: synthetic samples number; st: semantic_type; nt:noise_type
		self.model_name_suffix = '{}_lw@{}_nl@{}_sn@{}_st@{}_nt@{}.pth'.format(self.args.loss_type,
		                                                                       self.args.new_loss_weight,
		                                                                       self.args.noiseLen, self.args.n_samples,
		                                                                       self.args.semantic_type,
		                                                                       self.args.noise_type
		                                                                       )
	
	# Train teacher model
	def train_teacher(self, model, epochs, optimizer, criterion, train_data, test_data, model_save=True,
	                  draw_chart=False):
		model.to(self.device)
		lr_scheduler = StepLR(optimizer, step_size=15, gamma=0.9)
		_max = 0
		line_graph_data = []
		"""
		add dp to teacher
		"""
		if not self.args.disable_dp:
			model = convert_batchnorm_modules(model)
			model = model.to(self.device)

			privacy_engine = PrivacyEngine(
				model,
				batch_size=self.args.batch_size,
				sample_size=len(train_data),
				#alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
				alpha = range(2, 32),
				noise_multiplier=self.args.sigma,
				max_grad_norm=self.args.max_per_sample_grad_norm,
			)
			privacy_engine.attach(optimizer)
			#print('using sigma:{}'.format(optimizer.noise_multiplier))
		for epoch in range(epochs):
			model.train()
			_loss = 0.0
			for i, (images, labels) in enumerate(train_data, 0):
				predicts = model(Variable(images))
				labels = labels.squeeze()
				loss = criterion(predicts, labels.long())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				_loss += loss.item()
			# Test the model and save the model
			model.eval()
			test_result = self.test_model_or_filter_samples([test_data], teacher_or_student=model)
			if self.args.save_lgd and (epoch + 1) % 5 == 0:
				line_graph_data.append([epoch + 1, round(test_result, 2)])
			if test_result >= _max:
				_max = test_result
				if model_save:
					self.save_model(model, 'teacher')
			if not self.args.disable_dp:
				epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(self.args.delta)
				print(
					f"Train Epoch: {epoch} \t"
					f"Loss: {_loss:.6f} "
					f"(ε = {epsilon:.2f}, δ = {self.args.delta}) for α = {best_alpha}"
				)
			print(
				'| No.{} epoch | loss: {:.2f} | current_accuracy: {:.2f}% | top_accuracy:{:.2f}% |'.format(epoch,
				                                                                                           _loss,
				                                                                                           test_result,
				                                                                                           _max))
			lr_scheduler.step()
		if self.args.save_lgd:
			self.pro_data.store_res_for_line_graph(line_graph_data, 'teacher')
		return _max
	
	# Train generator model
	def train_generator(self, generator, teacher, epochs, optimizer, criterion, train_data, val_data,
	                    model_save=True,
	                    center=None):
		generator.to(self.device)
		teacher.eval()
		teacher.requires_grad = False
		lr_scheduler = StepLR(optimizer, step_size=15, gamma=0.9)
		test_max_acc = 0.00
		for epoch in range(epochs):
			_loss = 0.00
			tot_acc = []
			top_acc = 0.00
			generator.train()
			teacher.eval()
			for (features, labels) in train_data:
				fake_features = generator(Variable(features))
				output = teacher(fake_features)
				
				loss = criterion(output, labels.long())
				if self.args.loss_type == 'normal_loss':
					loss = loss
				elif self.args.loss_type == 'kl_loss':
					loss = criterion(teacher(fake_features), labels.long())
					kl_loss = self.new_loss.kl_loss(fake_features)
					loss = self.args.new_loss_weight * kl_loss + loss
				elif self.args.loss_type == 'mmd_loss':
					loss = criterion(teacher(fake_features), labels.long())
					mmd_loss = self.new_loss.mmd_loss(fake_features)
					loss = self.args.new_loss_weight * mmd_loss + loss
				elif self.args.loss_type == 'ct_loss':
					output.detach_()
					input_center = output
					input_center = Variable(input_center, requires_grad=True)
					# target_center = torch.zeros(1, input_center.size(1)).to(self.device)
					for i, j in enumerate(labels, 0):
						target_center = center[j].view(1, -1) if i == 0 else torch.cat(
							(target_center, center[j].view(1, -1)))
					center_loss = self.MSE_loss(input_center, target_center)
					# nor_loss = criterion(output1, labels.long())
					# center_loss = self.center_loss(input_center, labels, center)
					loss = self.args.new_loss_weight * center_loss + loss
				optimizer.zero_grad()
				# optimizer_c.zero_grad()
				loss.backward()
				optimizer.step()
				# optimizer_c.step()
				_loss += loss.item()
				# Calculate accuracy
				acc = self.test_model_or_filter_samples([fake_features, labels], teacher_or_student=teacher)
				tot_acc.append(acc)
			ave_acc = np.mean(tot_acc)
			if ave_acc >= top_acc:
				top_acc = ave_acc
				if model_save:
					self.save_model(generator, 'generator')
			
			generator.eval()
			test_result = self.test_model_or_filter_samples([val_data], teacher_or_student=teacher,
			                                                generator=generator)
			if test_result > test_max_acc:
				test_max_acc = test_result
			# if model_save:
			#     self.save_model(generator, 'generator')
			
			# TODO: split data and test generator
			print(
				"| No.{} epoch | loss: {:.2f} | train_cur_acc:{:.2f}% | train_top_acc:{:.2f}% | test_cur_acc:{"
				":.2f}% | test_top_acc:{:.2f}% |".format(
					epoch, _loss, ave_acc, top_acc, test_result, test_max_acc))
			
			lr_scheduler.step()
		return test_max_acc
	
	# Train Student model
	def train_student(self, student, epochs, s_optimizer, criterion, train_data, val_data, test_data,
	                  generator=None, g_optimizer=None, teacher=None, model_save=True):
		if generator is not None and teacher is not None:  # black box
			generator.to(self.device)
			teacher.to(self.device)
			teacher.requires_grad = False
		# if not self.args.disable_dp:
			#student = convert_batchnorm_modules(student)
		student.to(self.device)
		lr_s_scheduler = StepLR(s_optimizer, step_size=15, gamma=0.9)
		if generator is not None and teacher is not None:
			lr_g_scheduler = StepLR(g_optimizer, step_size=15, gamma=0.9)
		_max = 0
		_max_ = 0
		line_graph_data = []
		for epoch in range(epochs):
			student.train()
			if generator is not None and teacher is not None:  # black_box
				generator.train()
				teacher.eval()
			_loss = 0.00
			for i, (features, labels) in enumerate(train_data, 0):
				if generator is not None and teacher is not None:  # black_box
					features = generator(Variable(features)).data
					labels = f.softmax(teacher(Variable(features)), dim=1).data
				output = f.softmax(student(Variable(features)), dim=1)
				if generator is not None and teacher is not None:  # black_box
					loss = criterion(output, Variable(labels, requires_grad=True))
					if self.args.loss_type == 'normal_loss':
						loss = loss
					elif self.args.loss_type == 'kl_loss':
						kl_loss = self.new_loss.kl_loss(features)
						loss = self.args.new_loss_weight * kl_loss + loss
					elif self.args.loss_type == 'mmd_loss':
						mmd_loss = self.new_loss.mmd_loss(features)
						loss = self.args.new_loss_weight * mmd_loss + loss
				else:
					loss = criterion(output, Variable(labels, requires_grad=False))
				s_optimizer.zero_grad()
				if generator is not None and teacher is not None:  # black_box
					g_optimizer.zero_grad()
				loss.backward()
				s_optimizer.step()
				if generator is not None and teacher is not None:  # black_box
					g_optimizer.step()
				_loss += loss.item()
			student.eval()
			if generator is not None and teacher is not None:  # black_box
				generator.eval()
			if generator is not None and teacher is not None:  # black_box
				val_result = self.test_model_or_filter_samples([val_data], teacher_or_student=student,
				                                               generator=generator)
			else:
				val_result = self.test_model_or_filter_samples([val_data], teacher_or_student=student)
			test_result = self.test_model_or_filter_samples([test_data], teacher_or_student=student)
			if self.args.save_lgd and (epoch + 1) % 5 == 0:
				line_graph_data.append([epoch + 1, round(test_result, 2)])
			if test_result > _max_:
				_max_ = test_result
			if val_result >= _max:
				_max = val_result
				if model_save:
					self.save_model(student, 'student')
					if generator is not None and teacher is not None:
						self.save_model(generator, 'generator')
			
			print(
				'| No.{} epoch | loss: {:.2f} | cur_acc(fake_data): {:.2f}% | top_acc(fake_data):{:.2f}% | cur_acc('
				'real_data):{:.2f}% | top_acc(real_data):{:.2f}% |'.format(epoch, _loss, val_result, _max,
				                                                           test_result,
				                                                           _max_))
			lr_s_scheduler.step()
			if generator is not None and teacher is not None:  # black_box
				lr_g_scheduler.step()
		if self.args.save_lgd:
			self.pro_data.store_res_for_line_graph(line_graph_data, 'student')
		return _max, _max_
	
	def train_z_net(self, z_net, generator, epochs, optimizer, criterion, train_data, test_data, model_save=True):
		z_net.to(self.device)
		generator.to(self.device)
		generator.requires_grad = False
		lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.9)
		tot_acc = []
		_max = 0.00
		_max_ = 0.00
		for epoch in range(epochs):
			z_net.train()
			generator.eval()
			_loss = 0.00
			for (features, labels) in train_data:
				fake_features = generator(Variable(features))
				predicts = z_net(fake_features)
				labels = labels.long() if labels.ndim == 1 else labels.squeeze().long()
				loss = criterion(predicts, labels)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				_loss += loss.item()
				val_acc = self.test_model_or_filter_samples([fake_features, labels], teacher_or_student=z_net)
				tot_acc.append(val_acc)
			ave_acc = np.mean(tot_acc)
			if ave_acc >= _max:
				_max = ave_acc
				if model_save:
					self.save_model(z_net, 'z_net')
			z_net.eval()
			test_acc = self.test_model_or_filter_samples([test_data], teacher_or_student=z_net)
			if test_acc > _max_:
				_max_ = test_acc
			print(
				"| No.{} epoch | loss: {:.2f} | train_cur_acc:{:.2f}% | train_top_acc:{:.2f}% | test_cur_acc:{"
				":.2f}% | test_top_acc:{:.2f}% |".format(
					epoch, _loss, ave_acc, _max, test_acc, _max_))
			lr_scheduler.step()
	
	# 1.Test model(Teacher/Generator/Student model) 2.Quality check: return filtered samples for training student model
	def test_model_or_filter_samples(self, *dataset, teacher_or_student=None, generator=None,
	                                 need_quality_check='test_g',
	                                 test_by_categories=False, seen_or_unseen='seen'):
		total_img = 0
		correct_img = 0
		mean_acc = 0.00
		right_idx = []
		right_features = torch.zeros(1, self.args.resSize).to(self.device)
		right_softmax = torch.zeros(1, self.args.outSizeTS).to(self.device)
		right_labels = torch.zeros(1, ).to(self.device)
		predict_labels = torch.zeros(1, ).to(self.device)
		real_labels = torch.zeros(1, ).to(self.device)
		
		def calculate_accuracy(images, labels, need_generator=False):
			nonlocal total_img, correct_img, predict_labels, real_labels, mean_acc
			teacher_or_student.requires_grad = False
			if need_generator:
				generator.requires_grad = False
				images = generator(Variable(images, requires_grad=False))
			output = f.softmax(teacher_or_student(images), dim=1)
			_, predict = torch.max(output, 1)
			predict = predict.to(torch.int32)
			labels = labels if labels.ndim == 1 else labels.squeeze()
			total_img += labels.shape[0]
			# noinspection PyTypeChecker
			correct_img += torch.sum(predict == labels).item()
			if test_by_categories:
				start_num = 1 if predict_labels.size(0) == 1 else 0
				if predict_labels.size(0) == 1:
					real_labels = labels[0].detach().clone().view(-1, )
					predict_labels = predict[0].detach().clone().view(-1, )
				for i in range(start_num, labels.size(0)):
					real_labels = torch.cat((real_labels, labels[i].view(-1, )), 0)
					predict_labels = torch.cat((predict_labels, predict[i].view(-1, )), 0)
			if need_quality_check == 'quality':
				quality_check(images, output, predict, labels)
			elif need_quality_check == 'no_quality':
				images, output, labels = images.unsqueeze(1), output.unsqueeze(1), labels.unsqueeze(1)
				save_syn_data(images, output, labels)
		
		def save_syn_data(features, softmax, labels):
			nonlocal right_idx, right_features, right_softmax, right_labels
			start_num = 1 if right_features.size(0) == 1 else 0
			if right_features.size(0) == 1:
				right_features = features[0]
				right_softmax = softmax[0]
				right_labels = labels[0]
			for i in range(start_num, features.size(0)):
				right_features = torch.cat((right_features, features[i]), 0)
				right_softmax = torch.cat((right_softmax, softmax[i]), 0)
				right_labels = torch.cat((right_labels, labels[i]), 0)
		
		def quality_check(features, softmax, pre_labels, rea_labels):
			# nonlocal right_idx, right_features, right_softmax, right_labels
			correct_labels = (pre_labels == rea_labels)
			# noinspection PyTypeChecker
			batch_correct_idx = torch.nonzero(correct_labels == 1)
			num_correct_samples = batch_correct_idx.shape[0]
			if num_correct_samples > 0:
				right_idx.append(batch_correct_idx)
				r_feature = features[batch_correct_idx]
				r_softmax = softmax[batch_correct_idx]
				r_label = rea_labels[batch_correct_idx]
				save_syn_data(r_feature, r_softmax, r_label)
		
		if len(dataset[0]) == 1:
			data = dataset[0][0]
		elif len(dataset[0]) == 2:
			images, labels = dataset[0][0], dataset[0][1]
		if teacher_or_student is not None and generator is None:  # test student/Znet model
			teacher_or_student.eval()
			if len(dataset[0]) == 1:
				for (img, label) in data:
					calculate_accuracy(img, label, need_generator=False)
				if test_by_categories:
					seen_cls, unseen_cls = self.pro_data.obtain_classes()
					target_cls = seen_cls if seen_or_unseen == 'seen' else unseen_cls
					mean_acc = self.compute_per_class_acc(real_labels, predict_labels, target_cls) * 100
			elif len(dataset[0]) == 2:
				calculate_accuracy(images, labels, need_generator=False)
			accuracy = (correct_img / total_img) * 100
			accuracy = mean_acc if test_by_categories else accuracy
			return accuracy
		if teacher_or_student is not None and generator is not None and need_quality_check == 'test_g':  # test generator model
			generator.eval()
			if len(dataset[0]) == 1:
				for (img, label) in data:
					calculate_accuracy(img, label, need_generator=True)
			elif len(dataset[0]) == 2:
				calculate_accuracy(images, labels, need_generator=False)
			accuracy = (correct_img / total_img) * 100
			return accuracy
		if teacher_or_student is not None and generator is not None and (
				need_quality_check == 'quality' or need_quality_check == 'no_quality'):  # quality check
			teacher_or_student.eval()
			generator.eval()
			for (img, label) in data:
				calculate_accuracy(img, label, need_generator=True)
			q_accuracy = (correct_img / total_img) * 100
			tot_right_samples = right_features.shape[0]
			train_indices, val_indices = self.pro_data.split_data_by_proportion(tot_right_samples)
			train_features = right_features[train_indices]
			train_softmax = right_softmax[train_indices]
			val_features = right_features[val_indices]
			val_label = right_labels[val_indices]
			train_dataloader = self.pro_data.create_dataloader(train_features, train_softmax, self.args.batchTS)
			val_dataloader = self.pro_data.create_dataloader(val_features, val_label, self.args.batchTS)
			return train_dataloader, val_dataloader, q_accuracy
	
	# calculate per-class accuracy
	def compute_per_class_acc(self, real_labels, predict_labels, target_classes):
		target_classes = self.pro_data.realign_labels(
			target_classes) if self.args.AZSL_test == 'zsl' else target_classes
		acc_per_classes = []
		for i in target_classes:
			idx = torch.eq(real_labels, i)
			right_num = torch.sum(real_labels[idx] == predict_labels[idx]).item()
			num_per_cls = torch.sum(idx).item()
			acc_now_class = right_num / num_per_cls if num_per_cls != 0 else 0
			acc_per_classes.append(acc_now_class)
		mean_acc = np.mean(acc_per_classes)
		return mean_acc
	
	# Obtain dataset for train/test/val in teacher/generator/student models
	def obtain_dataset(self, model_name, teacher=None, generator=None):
		teacher_indices = {'a': 'train_allclass_loc',
		                   'b': 'test_allclass_loc'} if self.args.task_categories == 'GZSL_all' else {
			'a': 'trainval_loc', 'b': 'test_seen_loc'}
		student_indices = {'a': 'test_allclass_loc'} if self.args.task_categories == 'GZSL_all' else {
			'a': 'test_seen_loc'}
		z_net_indices = {'a': 'test_unseen_loc'} if self.args.AZSL_test == 'zsl' else {'a': 'test_unseen_loc',
		                                                                               'b': 'test_seen_loc'}
		generator_student_train_we = self.all_cls_wb if self.args.task_categories == 'GZSL_all' else self.train_cls_wb
		z_net_train_we = self.test_cls_wb if self.args.AZSL_test == 'zsl' else self.all_cls_wb
		need_realign_label = False if self.args.task_categories == 'GZSL_all' else True
		
		if model_name == 'teacher':
			train_features, train_labels, val_features, val_labels = self.pro_data.cuda_data(
				self.pro_data.split_data_by_indices(**teacher_indices))
			train_data_loader = self.pro_data.create_dataloader(train_features, train_labels, self.args.batchTS,
			                                                    need_realignment=False)
			val_data_loader = self.pro_data.create_dataloader(val_features, val_labels, self.args.batchTS,
			                                                  need_realignment=False)
			return train_data_loader, val_data_loader
		
		elif model_name == 'generator':
			train_test_features, train_test_labels = self.pro_data.create_repetitive_samples_for_generator(
				generator_student_train_we,
				self.args.semantic_type,
				self.args.n_samples,
				self.args.att_tc, self.args.outSizeTS)
			tot_samples_num = train_test_features.shape[0]
			train_indices, val_indices = self.pro_data.split_data_by_proportion(tot_samples_num)
			#test_features = torch.take(train_test_features, torch.tensor(val_indices).cuda())
			#test_labels = torch.take(train_test_labels, torch.tensor(val_indices).cuda())
			
			#train_features = torch.take(train_test_features, torch.tensor(train_indices).cuda())
			#train_labels = torch.take(train_test_labels, torch.tensor(train_indices).cuda())
			
			train_features, train_labels = train_test_features[train_indices], train_test_labels[train_indices]
			test_features, test_labels = train_test_features[val_indices], train_test_labels[val_indices]
			train_data_loader = self.pro_data.create_dataloader(train_features, train_labels, self.args.batchTS,
			                                                    need_realignment=False)
			test_data_loader = self.pro_data.create_dataloader(test_features, test_labels, self.args.batchTS,
			                                                   need_realignment=False)
			return train_data_loader, test_data_loader
		
		elif model_name == 'student':
			assert teacher is not None and generator is not None, 'Please pass in the pre-train teacher model and ' \
			                                                      'generator model '
			train_val_features, train_val_labels = self.pro_data.create_repetitive_samples_for_generator(
				generator_student_train_we,
				self.args.semantic_type,
				self.args.n_samples,
				self.args.att_tc, self.args.outSizeTS)
			train_val_data_loader = self.pro_data.create_dataloader(train_val_features, train_val_labels,
			                                                        self.args.batchTS,
			                                                        need_realignment=False)
			test_features, test_labels = self.pro_data.cuda_data(
				self.pro_data.split_data_by_indices(**student_indices))
			test_dataloader = self.pro_data.create_dataloader(test_features, test_labels, self.args.batchTS,
			                                                  need_realignment=False)
			# Split train dataset and validation dataset according the framework
			if self.args.framework == 'white_box':
				train_dataloader, val_dataloader, quality_accuracy = self.test_model_or_filter_samples(
					[train_val_data_loader],
					teacher_or_student=teacher,
					generator=generator,
					need_quality_check=self.args.need_quality_check)
				return train_dataloader, val_dataloader, test_dataloader, quality_accuracy
			elif self.args.framework == 'black_box':
				tot_right_samples = train_val_features.shape[0]
				train_indices, val_indices = self.pro_data.split_data_by_proportion(tot_right_samples)
				train_features = train_val_features[train_indices]
				train_labels = train_val_labels[train_indices]
				val_features = train_val_features[val_indices]
				val_label = train_val_labels[val_indices]
				train_dataloader = self.pro_data.create_dataloader(train_features, train_labels, self.args.batchTS)
				val_dataloader = self.pro_data.create_dataloader(val_features, val_label, self.args.batchTS)
				return train_dataloader, val_dataloader, test_dataloader
		
		elif model_name == 'z_net':
			train_features, train_labels = self.pro_data.create_repetitive_samples_for_generator(
				z_net_train_we,
				self.args.semantic_type,
				self.args.n_z_samples,
				'zsl_unseen', self.args.outSizeZ)
			if self.args.AZSL_test == 'zsl':
				test_features, test_labels = self.pro_data.cuda_data(
					self.pro_data.split_data_by_indices(**z_net_indices))
				train_dataloader = self.pro_data.create_dataloader(train_features, train_labels, self.args.batchTS,
				                                                   need_realignment=True)
				test_dataloader = self.pro_data.create_dataloader(test_features, test_labels, self.args.batchTS,
				                                                  need_realignment=True)
				return train_dataloader, test_dataloader
			if self.args.AZSL_test == 'gzsl':
				test_unseen_features, test_unseen_labels, test_seen_features, test_seen_labels = self.pro_data.cuda_data(
					self.pro_data.split_data_by_indices(**z_net_indices))
				train_dataloader = self.pro_data.create_dataloader(train_features, train_labels, self.args.batchTS)
				
				test_features = torch.cat((test_unseen_features, test_seen_features), dim=0)
				test_labels = torch.cat((test_unseen_labels, test_seen_labels), dim=0)
				
				test_unseen_dataloader = self.pro_data.create_dataloader(test_unseen_features, test_unseen_labels,
				                                                         self.args.batchTS)
				test_seen_dataloader = self.pro_data.create_dataloader(test_seen_features, test_seen_labels,
				                                                       self.args.batchTS)
				test_dataloader = self.pro_data.create_dataloader(test_features, test_labels,
				                                                  self.args.batchTS)
				return train_dataloader, test_seen_dataloader, test_unseen_dataloader, test_dataloader
		else:
			print('Wong parameter! Available parameter contain "teacher", "generator", "student"')
	
	# Load pre-train model based on the model name, available parameters contain: 'teacher', 'generator', 'student'
	def load_model(self, model_name):
		model = self.__getattribute__(model_name)
		if not self.args.disable_dp and model_name=='teacher':
			model = convert_batchnorm_modules(model)
		model_path = os.path.join(self.save_path,
		                          '{}_{}'.format(model_name, self.model_name_suffix))
		
		model.load_state_dict(torch.load(model_path))
		model.to(self.device)
		return model
	
	# Save models
	def save_model(self, model, model_name):
		if not os.path.exists(self.save_path):
			os.makedirs(self.save_path)
		torch.save(model.state_dict(), os.path.join(self.save_path,
		                                            '{}_{}'.format(model_name, self.model_name_suffix)))
	
	# Load the best model
	def load_best_model(self, model_name):
		def match(patt_a, patt_b, mn):
			return patt_a.search(mn) and patt_b.search(mn)
		
		model = self.__getattribute__(model_name)
		model_path = self.best_model_path
		model_list = os.listdir(model_path)
		best_model_result = str(max([float(model.split("@")[1]) for model in model_list]))
		re_model_name, best_model_result = re.compile(model_name), re.compile(best_model_result)
		model_index = np.array([isinstance(i, re.Match) for i in
		                        list(map(lambda x: match(re_model_name, best_model_result, x), model_list))])
		best_model_path = os.path.join(model_path, model_list[np.where(model_index == True)[0][0]])
		print('\n', '*' * 160, '\n', 'Current {} model is {} '.format(model_name, best_model_path), '\n', '*' * 160)
		model.load_state_dict(torch.load(best_model_path))
		model.to(self.device)
