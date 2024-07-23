import datetime
import sys
import os
import numpy as np
import pandas as pd
import torch

from args import Args
from pro_data import ProData
from train import Training
import models as models

# sys.path.append("libs")
# sys.path.append("checkpoints")
args = Args().parse()
pro_data = ProData(args)
test_data = [[5, 85.0], [10, 90.1]]
pro_data.store_res_for_line_graph(test_data, 'teacher')



# save_path = os.path.join(args.checkPointRoot, args.framework, args.task_categories)
# save_path = save_path if args.task_categories == 'GZSL_all' else os.path.join(save_path,args.AZSL_test)
#
# save_path = os.path.join(save_path, args.dataset)
# model_path = os.path.join(save_path, '{}_{}.pth'.format('teacher', args.noiseLen))
# teacher = models.Teacher(args.resSize, args.hidSizeTSZ_1, args.hidSizeTSZ_2, args.outSizeTS, args.drop_p)
# teacher.load_state_dict(torch.load(model_path))
# print(teacher.bn1.running_mean.size())
# print(teacher.bn1.running_mean)
# print(teacher.bn1.running_var.size())
# print(teacher.bn1.running_var)