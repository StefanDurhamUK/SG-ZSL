import datetime
import os
import sys
import numpy as np
import pandas as pd
import torch
import gc
from pylint.checkers import variables

from args import Args
from pro_data import ProData
from train import Training
from new_loss import NewLoss
from experiments_results import ExperimentResults
import warnings
warnings.filterwarnings("ignore")

# Root setting
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

# Initializations
args = Args().parse()
keywords = args.query_keywords
train = Training(args)
pro_data = ProData(args)
n_loss = NewLoss()
exp_res = ExperimentResults()

# Initialize time
time = datetime.datetime.now()
cur_time = str(time.month) + '-' + str(time.day) + ' ' + str(time.hour) + ':' + str(time.minute) + ':' + str(
    time.second)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

# Print prompt information
print('\n', '=' * 30,
      ' | Experiments on the {} dataset under {} / {} / {} / lw:{} / dt:{} / nl:{} | '
      .format(args.dataset, args.framework, args.task_categories, args.loss_type, args.new_loss_weight, args.drop_p,
              args.noiseLen), '=' * 30, '\n')

# TRAIN
if args.train:
    # Train teacher model
    if args.trainTeacherNet:
        print('\n', '*' * 30, ' | Train Teacher Model | ', '*' * 30, '\n')
        teacher_train_dataloader, teacher_test_dataloader = train.obtain_dataset(model_name='teacher')
        teacher_acc = train.train_teacher(train.teacher, args.epochT, train.T_opti, train.CRE_loss,
                                          teacher_train_dataloader, teacher_test_dataloader)

    if args.framework == 'white_box':
        # Train generator model
        if args.trainGeneratorNet:
            print('\n', '*' * 30, ' | Train Generator Model | ', '*' * 30, '\n')
            train.load_model('teacher')
            #train.load_best_model('teacher')
            # cls_center = n_loss.obtain_rf_center(train.teacher) if args.loss_type == "ct_loss" else None
            # bn_mean = train.generator.net
            generator_train_dataloader, generator_test_dataloader = train.obtain_dataset(model_name='generator')
            generator_acc = train.train_generator(train.generator, train.teacher, args.epochG, train.G_opti,
                                                  train.CRE_loss,
                                                  generator_train_dataloader, generator_test_dataloader,
                                                  center=None)
        # Train student model
        if args.trainStudentNet:
            # load pre-train teacher model and generator model
            train.load_model('teacher')
            train.load_model('generator')
            # Train student
            student_train_dataloader, student_val_dataloader, student_test_dataloader, q_accuracy = train.obtain_dataset(
                model_name='student', teacher=train.teacher, generator=train.generator)
            tr_num = student_train_dataloader.sampler.num_samples
            va_num = student_val_dataloader.sampler.num_samples
            te_num = student_test_dataloader.sampler.num_samples
            if args.need_quality_check == 'no_quality':
                print('\n', '*' * 30, ' | Train Student Model | No Quality Check |'.format(q_accuracy),
                      'train/test/val:{}/{}/{} |'.format(tr_num, va_num, te_num), '*' * 30, '\n')
            else:
                print('\n', '*' * 30, ' | Train Student Model | quality rate:{:.2f}% |'.format(q_accuracy),
                      'train/test/val:{}/{}/{} |'.format(tr_num, va_num, te_num), '*' * 30, '\n')
            student_acc_fake, student_acc_real = train.train_student(train.student, args.epochS, train.S_opti,
                                                                     train.MSE_loss,
                                                                     student_train_dataloader, student_val_dataloader,
                                                                     student_test_dataloader)
    elif args.framework == 'black_box':
        if args.trainGeneratorNet and args.trainStudentNet:
            # load pre-train teacher model and generator model
            #train.load_model('teacher')
            train.load_best_model('teacher')
            # Train student
            student_train_dataloader, student_val_dataloader, student_test_dataloader = train.obtain_dataset(
                model_name='student', teacher=train.teacher, generator=train.generator)
            tr_num = student_train_dataloader.sampler.num_samples
            va_num = student_val_dataloader.sampler.num_samples
            te_num = student_test_dataloader.sampler.num_samples
            print('\n', '*' * 30, ' | Train Student Model | No quality check until now |',
                  'train/test/val:{}/{}/{} |'.format(tr_num, va_num, te_num), '*' * 30, '\n')
            args.epochS = args.epochS1
            args.s_weight_decay = 100
            student_acc_fake, student_acc_real = train.train_student(train.student, args.epochS, train.S_opti,
                                                                     train.MSE_loss,
                                                                     student_train_dataloader, student_val_dataloader,
                                                                     student_test_dataloader, generator=train.generator,
                                                                     g_optimizer=train.G_opti, teacher=train.teacher)
            if args.need_quality_check == 'quality':
                torch.cuda.empty_cache()
                del variables
                gc.collect()
                train.load_model('generator')
                args.framework = 'white_box'
                args.n_samples = args.bn_samples
                # Train student
                student_train_dataloader, student_val_dataloader, student_test_dataloader, q_accuracy = train.obtain_dataset(
                    model_name='student', teacher=train.teacher, generator=train.generator)
                tr_num = student_train_dataloader.sampler.num_samples
                va_num = student_val_dataloader.sampler.num_samples
                te_num = student_test_dataloader.sampler.num_samples
                args.epochS = args.epochS2
                print('\n', '*' * 30, ' | Train Student Model | quality rate:{:.2f}% |'.format(q_accuracy),
                      'train/test/val:{}/{}/{} |'.format(tr_num, va_num, te_num), '*' * 30, '\n')
                student_acc_fake, student_acc_real = train.train_student(train.student, args.epochS, train.S_opti,
                                                                         train.MSE_loss,
                                                                         student_train_dataloader,
                                                                         student_val_dataloader,
                                                                         student_test_dataloader)
                args.framework = 'black_box'

# TEST
if args.test:
    if args.task_categories == 'AZSL':
        #50 classes
        if args.AZSL_test == 'gzsl':
            if args.trainZNet:
                # Load pre-train teacher model and generator model
                train.load_model('generator')
                # Train Z model
                z_train_dataloader, z_test_seen_dataloader, z_test_unseen_dataloader, z_test_dataloader = \
                    train.obtain_dataset(model_name='z_net')
                tr_num = z_train_dataloader.sampler.num_samples
                te_seen_num = z_test_seen_dataloader.sampler.num_samples
                te_unseen_num = z_test_unseen_dataloader.sampler.num_samples

                print('\n', '*' * 30,
                      ' | Train Z Model  | train/test_seen/test_unseen:{}/{}/{} |'.format(tr_num, te_seen_num,
                                                                                          te_unseen_num), '*' * 30,
                      '\n')
                train.train_z_net(train.z_net, train.generator, args.epochZ, train.Z_opti, train.CRE_loss,
                                  z_train_dataloader, z_test_dataloader)

                # Real unseen data -> Z_net -> acc_unseen
                train.load_model('z_net')
                acc_unseen = train.test_model_or_filter_samples([z_test_unseen_dataloader],
                                                                teacher_or_student=train.z_net,
                                                                test_by_categories=True, seen_or_unseen='unseen')
                acc_seen = train.test_model_or_filter_samples([z_test_seen_dataloader], teacher_or_student=train.z_net,
                                                              test_by_categories=True, seen_or_unseen='seen')
                H = 0 if acc_seen == 0 and acc_unseen == 0 else 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)

                print('\n', '*' * 40, ' | Test Result | ', '*' * 40, '\n',
                      '| Acc_seen_class:{:.2f}% | Acc_unseen_class:{:.2f}% |  H={:.2f} |'.format(acc_seen, acc_unseen,
                                                                                                 H))
        # 10 classes
        if args.AZSL_test == 'zsl':
            if args.trainZNet:
                # Load pre-train generator model
                # train.load_model('generator')
                train.load_best_model('generator')
                # Train Z model
                z_train_dataloader, z_test_dataloader = train.obtain_dataset(model_name='z_net')
                tr_num = z_train_dataloader.sampler.num_samples
                te_num = z_test_dataloader.sampler.num_samples
                print('\n', '*' * 30, 'Train Z Model  | train/test/:{}/{} |'.format(tr_num, te_num), '*' * 30, '\n')
                train.train_z_net(train.z_net, train.generator, args.epochZ, train.Z_opti, train.CRE_loss,
                                  z_train_dataloader, z_test_dataloader)
                # real unseen -> Z_net -> acc_unseen
                train.load_model('z_net')
                acc_unseen = train.test_model_or_filter_samples([z_test_dataloader], teacher_or_student=train.z_net,
                                                                test_by_categories=True, seen_or_unseen='unseen')
                print('\n', '*' * 40, ' | Test Result | ', '*' * 40, '\n',
                      ' | Acc_unseen_class:{:.2f}% |'.format(acc_unseen))
        # elif args.AZSL_test == 'gzsl':
        #     # Load pre-train teacher model
        #     train.load_best_model('teacher')
        #     # Test teacher model
        #     teacher_test_dataloader = train.obtain_dataset(model_name='z_net')
        #     acc_seen = train.test_model_or_filter_samples([teacher_test_dataloader], teacher_or_student=train.teacher,
        #                                                   test_by_categories=True, seen_or_unseen='seen')
        #     acc_unseen = train.test_model_or_filter_samples([teacher_test_dataloader], teacher_or_student=train.teacher,
        #                                                     test_by_categories=True, seen_or_unseen='unseen')
        #     H = 0 if acc_seen == 0 and acc_unseen == 0 else 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
        #     print('\n', '*' * 40, ' | Test Result | ', '*' * 40, '\n',
        #           '| Acc_seen_class:{:.2f}% | Acc_unseen_class:{:.2f}% |  H={:.2f} |'.format(acc_seen, acc_unseen, H))
        #
    elif args.task_categories == 'GZSL_all':
        train.load_model('student')
        # Obtain test dataset
        indices = {'a': 'tunseen_test_loc', 'b': 'test_seen_loc'}
        test_unseen_features, test_unseen_labels, test_seen_features, test_seen_labels = train.pro_data.cuda_data(
            train.pro_data.split_data_by_indices(**indices))
        test_seen_dataloader = train.pro_data.create_dataloader(test_seen_features, test_seen_labels, args.batchTS,
                                                                drop_last=False)
        test_unseen_dataloader = train.pro_data.create_dataloader(test_unseen_features, test_unseen_labels,
                                                                  args.batchTS, drop_last=False)
        acc_seen = train.test_model_or_filter_samples([test_seen_dataloader], teacher_or_student=train.student,
                                                      test_by_categories=True, seen_or_unseen='seen')
        acc_unseen = train.test_model_or_filter_samples([test_unseen_dataloader],
                                                        teacher_or_student=train.student,
                                                        test_by_categories=True, seen_or_unseen='unseen')
        H = 0 if acc_seen == 0 and acc_unseen == 0 else 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
        print('\n', '*' * 40, ' | Test Result | ', '*' * 40, '\n',
              '| Acc_seen_class:{:.2f}% | Acc_unseen_class:{:.2f}% |  H={:.2f} |'.format(acc_seen, acc_unseen, H))
end.record()
# elif args.AZSL_test == 'gzsl':
# Load pre-train teacher model
#         train.load_best_model('teacher')
#         indices = {'a': 'tunseen_test_loc', 'b': 'test_seen_loc'}
#         test_unseen_features, test_unseen_labels, test_seen_features, test_seen_labels = train.pro_data.cuda_data(
#             train.pro_data.split_data_by_indices(**indices))
#         test_seen_dataloader = train.pro_data.create_dataloader(test_seen_features, test_seen_labels, args.batchTS,
#                                                                 drop_last=False)
#         test_unseen_dataloader = train.pro_data.create_dataloader(test_unseen_features, test_unseen_labels,
#                                                                   args.batchTS, drop_last=False)
#         acc_seen = train.test_model_or_filter_samples([test_seen_dataloader], teacher_or_student=train.teacher,
#                                                       test_by_categories=True, seen_or_unseen='seen')
#         acc_unseen = train.test_model_or_filter_samples([test_unseen_dataloader],
#                                                         teacher_or_student=train.teacher,
#                                                         test_by_categories=True, seen_or_unseen='unseen')
#         H = 0 if acc_seen == 0 and acc_unseen == 0 else 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
#         print('\n', '*' * 40, ' | Test Result | ', '*' * 40, '\n',
#               '| Acc_seen_class:{:.2f}% | Acc_unseen_class:{:.2f}% |  H={:.2f} |'.format(acc_seen, acc_unseen, H))
# end.record()
#         # Test teacher model
#         teacher_test_dataloader = train.obtain_dataset(model_name='z_net')
#         acc_seen = train.test_model_or_filter_samples([teacher_test_dataloader], teacher_or_student=train.teacher,
#                                                       test_by_categories=True, seen_or_unseen='seen')
#         acc_unseen = train.test_model_or_filter_samples([teacher_test_dataloader], teacher_or_student=train.teacher,
#                                                         test_by_categories=True, seen_or_unseen='unseen')
#         H = 0 if acc_seen == 0 and acc_unseen == 0 else 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
#         print('\n', '*' * 40, ' | Test Result | ', '*' * 40, '\n',
#               '| Acc_seen_class:{:.2f}% | Acc_unseen_class:{:.2f}% |  H={:.2f} |'.format(acc_seen, acc_unseen, H))

# Draw T-SNE graph
"""
问题1：目前数据比较少，理想情况下每个类中选出200-300个数据来画TSNE,但是目前有几个类数据严重不足，例如AWA1 第20类 第8类数据
解决方案一： 整个数据集都塞进去试一试
解决方案二： 类别少的类区全部的数据（<200）剩下的类别取（=200），
问题2： TSNE数据真假数据是否需要数量相等
"""
if args.do_tsne:
    teacher_indices = {'a': 'test_unseen_loc'} if args.tsne_data == 'test' else {'b': 'trainval_loc'}
    generator_indices = train.test_cls_wb if args.tsne_data == 'test' else train.train_cls_wb
    t_feature, t_labels, index = pro_data.obtain_original_data()
    train_idx = torch.LongTensor(index[str(list(teacher_indices.values())[0])].astype(np.int32).squeeze() - 1)
    train_features = torch.index_select(t_feature, 0, train_idx)
    train_labels = torch.index_select(t_labels, 0, train_idx).int().squeeze()
    teacher_features, teacher_labels = train_features.to('cpu').numpy(), train_labels.to('cpu').numpy()
    selected_teacher_features, selected_teacher_labels, marked_labels = pro_data.scale_data_by_cls_and_num(
        teacher_features,
        teacher_labels, args.tsne_cls_num, args.cls_samples_num)
    teacher_data = (selected_teacher_features, selected_teacher_labels)
    train.load_best_model('generator')
    # train.load_model('generator')
    generator_features, generator_labels = pro_data.create_repetitive_samples_for_generator(
        generator_indices,
        args.semantic_type,
        args.n_samples,
        'gzsl_seen', 50)
    generator_train_dataloader = pro_data.create_dataloader(generator_features, generator_labels, args.batchTS,
                                                            need_realignment=False)
    pro_data.draw_t_sne(teacher_data, generator_train_dataloader, train.generator, marked_labels)

# Calculate duration
torch.cuda.synchronize()
duration_time = round((start.elapsed_time(end) / 1000), 2)

# Check previous best result
best_result = exp_res.check_best_model(args.dataset)

# Save detailed experiment result to excel file.
if duration_time != 0.0 and args.save_result_in_excel:
    duration_time = str(duration_time) + 's'
    results_data = pd.DataFrame({'Time': cur_time,
                                 'Duration': duration_time,
                                 'Framework': args.framework,
                                 'Task_categories': args.task_categories,
                                 'AZSL_test': args.AZSL_test if args.task_categories == 'AZSL' else '',
                                 'Dataset': args.dataset,
                                 'Teacher_acc': '' if 'teacher_acc' not in dir() else str(round(teacher_acc, 2)) + '%',
                                 'Generator_acc': '' if 'generator_acc' not in dir() else str(
                                     round(generator_acc, 2)) + '%',
                                 'Quality_check_acc': '' if 'q_accuracy' not in dir() else str(
                                     round(q_accuracy, 2)) + '%',
                                 'Student_acc(fake_data)': '' if 'student_acc_fake' not in dir() else str(
                                     round(student_acc_fake, 2)) + '%',
                                 'Student_acc(real_data)': '' if 'student_acc_real' not in dir() else str(
                                     round(student_acc_real, 2)) + '%',
                                 'Acc_seen': '' if 'acc_seen' not in dir() else str(round(acc_seen, 2)) + '%',
                                 'Acc_unseen': '' if 'acc_unseen' not in dir() else str(round(acc_unseen, 2)) + '%',
                                 'H': '' if 'H' not in dir() else str(round(H, 2)),
                                 'Training_details': '| lrT:' + str(args.lrT) + ' | lrG:' + str(
                                     args.lrG) + ' | lrS:' + str(
                                     args.lrS) + ' | lrZ:' + str(
                                     args.lrZ) + ' |' + ' |' + 'nlt:' + str(args.loss_type) + ' |' + 'nlw:' + str(
                                     args.new_loss_weight) + ' |' + ' st:' + str(
                                     args.semantic_type) + ' |' + ' use_qc:' + str(
                                     args.need_quality_check) + ' |' + '\n' + '| nt:' + str(
                                     args.noise_type) + '| nl:' + str(
                                     args.noiseLen) + ' | nn:' + str(
                                     args.n_samples) + ' | ' + 'dt:' + str(args.drop_p) + ' | ' + 'Te:' + str(
                                     args.epochT) + ' | Ge:' + str(
                                     args.epochG) + ' | Se:' + str(args.epochS) + ' | Se1:' + str(
                                     args.epochS1) + ' | Se2:' + str(args.epochS2) + ' | Ze:' + str(
                                     args.epochZ) + ' |'},
                                index=[cur_time])
    exp_res.save_res_to_excel(results_data)
    # Print prompt information for historical top results
    args.query_keywords = {"keyword": "Dataset", "filter_by": args.dataset}
    exp_res.set_pd_options()
    print(results_data[args.prompt_info], '\n')

# Save the best model
if args.save_best_model and duration_time != 0.0:
    if not os.path.exists(train.best_model_path):
        os.makedirs(train.best_model_path)
    current_result = ["H", round(H, 2)] if args.task_categories == 'GZSL_all' or (
            args.task_categories == 'AZSL' and args.AZSL_test == 'gzsl') else ['U', round(acc_unseen, 2)]

    if best_result is None or current_result[1] > best_result:
        exp_res.move_best_model(train.save_path, train.best_model_path, train.model_name_suffix, current_result)
    performance_gap = current_result[1] - best_result if best_result is not None else 0
    print('*' * 40, ' | {}:improved by {} | '.format(current_result[0], round(performance_gap, 2)), '*' * 190, '\n',
          '\n', '\n', '=' * 30,
          ' | Complete the experiments on the {} dataset under {} / {} / {} / lw:{} / dt:{} / nl:{} | '
          .format(args.dataset, args.framework, args.task_categories, args.loss_type, args.new_loss_weight, args.drop_p,
                  args.noiseLen), '=' * 30, '\n')

else:
    print('\n', '*' * 30,
          '<Complete experiments on the {} dataset under {} / {} / {} / lw:{} / dt:{} / nl:{} | '
          .format(args.dataset, args.framework, args.task_categories, args.loss_type, args.new_loss_weight, args.drop_p,
                  args.noiseLen), '*' * 30, '\n')
