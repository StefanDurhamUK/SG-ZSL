import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as scio
import utils
import torch
import torch.utils.data as dt
from torch.autograd import Variable
from openTSNE import TSNE

COLOUR = ['DC143C', '800080', '0000FF', '00FFFF', '3CB371', 'FFD700', 'FF8C00', '8B0000', '000000',
		               'FF00FF', 'D2691E', '808080']


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
