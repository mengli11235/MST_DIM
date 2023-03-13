from cv2 import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import string
import glob
from scipy.stats import t
import pickle5 as pickle
import re
import copy
from atariari.methods.masked_stdim import MaskGenerator
import cv2
def plots(xs, ys, xlabel, ylabel, title, legends, loc="lower right", color=['b','y','g', 'r']):
    if not os.path.exists('figs'):
        os.makedirs('figs')
    for i,x in enumerate(xs):
        plt.plot(x,ys[i], linewidth=1.5,color=color[i],) #linestyle=(0, (i+3, 1, 2*i, 1)),)
    #plt.legend(loc=loc, ncol=1)
    #plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join('figs', title + ".pdf"))
    plt.close()

def plots_err(xs, ys, ystd, xlabel, ylabel, title, legends, loc="lower right", color=['b','y','g', 'r']):

    if not os.path.exists('figs'):
        os.makedirs('figs')
    for i,x in enumerate(xs):
        #plt.errorbar(x, ys[i], xerr=0.5, yerr=2*ystd[i], label=legends[i], color=color[i], linewidth=1.5,) #linestyle=(0, (i+3, 1, 2*i, 1)),)
        plt.plot(x,ys[i], color=color[i], linewidth=1.5,) #linestyle=(0, (i+3, 1, 2*i, 1)),)
        if True: #i==0:
            plt.fill_between(x, np.array(ys[i])-2*np.array(ystd[i]), np.array(ys[i])+2*np.array(ystd[i]), color=color[i], alpha=0.1)
    #plt.legend(loc=loc, ncol=1)
    #plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join('figs', title + ".pdf"))
    plt.close()

def find_checkpoint(base_path, postfix=''):
    model_dicts = {}
    game_names = []
    for path_to_load in sorted(glob.glob(base_path + '/*'), reverse=False):
        game_name = str(os.path.basename(path_to_load))
        game_names.append(game_name)
        model_dicts[game_name] = []
        for job_lib_file in sorted(glob.glob(path_to_load + '/' + postfix + '_e.log'), reverse=False):
            print(job_lib_file)
            with open(job_lib_file, "r") as fp:
                model_dict = fp.read()
                model_dicts[game_name].append(model_dict)
    return model_dicts, game_names

def summarize_by_cat(dict, subcat='across_categories_avg_acc'):
    summarized = {}
    mean_cat = 0
    num_games = len(dict.keys())
    for key in dict.keys():
        files = dict[key]
        categories = 0
        for f in files:
            x = re.search(subcat+'\s0.*', f)
            if x is not None:
                categories += float(x.group(0).split()[1])
            else:
                print(key + ' has no ' + subcat)
                categories = 0
                num_games -= 1
                break
        categories /= len(files)
        #print(categories)
        mean_cat += categories

        summarized[key] = categories

        # with open('sum/' + key + '.txt', 'w') as f:
        #     f.write(subcat + str(round(categories, 2)))
    print(str(num_games)+' out of ' + str(len(dict.keys())))
    summarized['mean'] = mean_cat/num_games
    return summarized

def summarize(dict, cat_list):
    dict_list = {}
    for c in cat_list:
        dict_list[c] = summarize_by_cat(dict, c)
    return dict_list

def print_table_by_cat(summarized, game_names, cat='across_categories_avg_acc'):
    game_names = copy.deepcopy(game_names)
    game_names.append('mean')
    str_out = 'Games&'
    for m in summarized.keys():
        str_out +=  string.capwords(m.replace("_", " "))+'&'
    str_out = str_out[:-1]
    str_out += '\\\\'
    for g in game_names:
        if g == 'mz_revenge':
            g_print = 'Montezuma Revenge'
        else:
            g_print = string.capwords(g.replace("_", " "))
        str_out += ('\\hline\n'+g_print+'&')
        for m in summarized.keys():
            if m == 'observable' or is_max(summarized, m, cat, g):
                str_out += '\\textbf{' + (str(round(summarized[m][cat][g],2)) + '}' +'&')
            else:
                str_out += (str(round(summarized[m][cat][g],2))+'&')
        str_out = str_out[:-1]
        str_out += '\\\\\n'

    with open('sum/table_' + cat + '.txt', 'w') as f:
        f.write(str_out)

def is_max(summarized, m, cat, g):
    is_max = True
    for m_ in summarized.keys():
        if m_ != 'observable' and round(summarized[m_][cat][g],2) > round(summarized[m][cat][g],2):
            is_max = False
            break
    return is_max

if __name__ == '__main__':
    mask_generator = MaskGenerator(mask_ratio=0.4)
    mask = mask_generator()
    img = cv2.imread('sum/img.png')
    imge =img.resize((210,160))
    print(imge.shape)
    plt.imshow(img*(1-mask))
    plt.savefig('masked.pdf')
    plt.close()

    base_path = 'res/'
    summarized = {}
    cat_list = ['across_categories_avg_acc', 'across_categories_avg_f1', 'agent_localization_avg_acc', 'agent_localization_avg_f1', 'small_object_localization_avg_acc', 'small_object_localization_avg_f1', 'score_clock_lives_display_avg_acc', 'score_clock_lives_display_avg_f1', 'misc_keys_avg_acc', 'misc_keys_avg_f1', 'other_localization_avg_acc', 'other_localization_avg_f1']

    ### basline
    base_dicts, game_names = find_checkpoint(base_path, postfix='*base')
    summarized['observable'] = summarize(base_dicts, cat_list=cat_list)
    #### probe with masks
    probe_dicts, _ = find_checkpoint(base_path, postfix='??')
    probe_dicts0, _ = find_checkpoint(base_path, postfix='?')
    for key in probe_dicts0.keys():
        probe_dicts[key].extend(probe_dicts0[key])
    summarized['non-observable'] = summarize(probe_dicts, cat_list=cat_list)

    #### supervised
    supervised_dicts, _ = find_checkpoint(base_path, postfix='*supervised')
    summarized['supervised'] = summarize(supervised_dicts, cat_list=cat_list)

    #### pretrain with masked images
    pretrain_dicts, _ = find_checkpoint(base_path, postfix='*pretrain')
    summarized['pretrain'] = summarize(pretrain_dicts, cat_list=cat_list)

    #### pretrain with mask ratio 0.2
    ratio2_dicts, _ = find_checkpoint(base_path, postfix='*ratio2')
    summarized['ratio_0.2'] = summarize(ratio2_dicts, cat_list=cat_list)
    #### pretrain with mask ratio 0.6
    ratio6_dicts, _ = find_checkpoint(base_path, postfix='*ratio6')
    summarized['ratio_0.6'] = summarize(ratio6_dicts, cat_list=cat_list)
    #### pretrain with masked ratio 0.8
    ratio8_dicts, _ = find_checkpoint(base_path, postfix='*ratio8')
    summarized['ratio_0.8'] = summarize(ratio8_dicts, cat_list=cat_list)
    #print(len(game_names))
    for c in cat_list:
        print_table_by_cat(summarized, game_names=game_names, cat=c)

    # game_name = 'krull'
    # titile_name = string.capwords(game_name.replace("_", " "))
    # path1 = '../swin_results/model_savedir/' + game_name + '00/'+game_name+'_bestq.pkl'
    # path2 = '../swin_results/model_savedir/' + game_name + '01/'+game_name+'_bestq.pkl'

    # model_dict1 = torch.load(path1, map_location=torch.device('cpu'))
    # model_dict2 = torch.load(path2, map_location=torch.device('cpu'))

    # legends = ['Swin DQN', 'Double DQN']
    # perf_range = np.arange(0, 8, 0.1)
    # perf_scores1 = np.zeros(len(perf_range))
    # perf_scores2 = np.zeros(len(perf_range))

    for i, model_dict1 in enumerate(base_dicts):
        pass
        # model_dict2 = model_dicts2[i]
        # game_name = game_names1[i]
        # assert game_name == game_names2[i]

        # info = model_dict1['info']
        # perf1 = model_dict1['perf']
        # perf2 = model_dict2['perf']
        # titile_name = string.capwords(game_name.replace("_", " "))

        # steps1 = perf1['steps']
        # steps2 = perf2['steps']
        # eval_steps1 = perf1['eval_steps']
        # eval_steps2 = perf2['eval_steps']

        # y1_mean_scores = perf1['eval_rewards']
        # y1_std_scores = perf1['eval_stds']
        # y1q = perf1['q_record']

        # y2_mean_scores = perf2['eval_rewards']
        # y2_std_scores = perf2['eval_stds']
        # y2q = perf2['q_record']


    # ## Mean Eval Normalized
        # mean_score1 = (y1_mean_scores[-1]-random_human[game_name][0])/(random_human[game_name][1]-random_human[game_name][0])

        # mean_score2 = (y2_mean_scores[-1]-random_human[game_name][0])/(random_human[game_name][1]-random_human[game_name][0])

        # print(titile_name,'&', round(y2_mean_scores[-1],2), '&', round(y2_std_scores[-1],2),'&', round(mean_score2,2),'&', round(y1_mean_scores[-1],2), '&' , round(y1_std_scores[-1],2), '&', round(mean_score1,2), '\\\\')
        # print('\\hline')

    ## Highest Eval Normalized
        # highest_score1 = (perf1['highest_eval_score'][-1]-random_human[game_name][0])/(random_human[game_name][1]-random_human[game_name][0])

        # highest_score2 = (perf2['highest_eval_score'][-1]-random_human[game_name][0])/(random_human[game_name][1]-random_human[game_name][0])

        # print(game_name, perf2['highest_eval_score'][-1], round(highest_score2,2), perf1['highest_eval_score'][-1], round(highest_score1, 2))

    # ## Performance Profiles
    #     samples1 = np.random.normal(y1_mean_scores[-1], y1_std_scores[-1], 100)
    #     normalized_samples1 = (samples1-random_human[game_name][0])*100/(random_human[game_name][1]-random_human[game_name][0])

    #     samples2 = np.random.normal(y2_mean_scores[-1], y2_std_scores[-1], 100)
    #     normalized_samples2 = (samples2-random_human[game_name][0])*100/(random_human[game_name][1]-random_human[game_name][0])

    #     for x in normalized_samples1:
    #         for i in range(len(perf_range)):
    #             if x >= perf_range[i]*100:
    #                 perf_scores1[i] += 1
    #             else:
    #                 break

    #     for x in normalized_samples2:
    #         for i in range(len(perf_range)):
    #             if x >= perf_range[i]*100:
    #                 perf_scores2[i] += 1
    #             else:
    #                 break



    ### AUC
        # auc1 = 0
        # auc2 = 0
        # auc_dqn = 0
        # for i in range (0, min(len(y1_mean_scores), len(y2_mean_scores)), 2):
        # #for i in range (int(min(len(y1_mean_scores), len(y2_mean_scores))/2), min(len(y1_mean_scores), len(y2_mean_scores))):
        #     auc1 += y1_mean_scores[i]
        #     auc2 += y2_mean_scores[i]
        #     auc_dqn += random_human[game_name][2]
        
        # print(game_name, auc1/abs(auc_dqn), auc2/abs(auc_dqn))

    ## Mean
        # title = "Mean Evaluation Scores in "+ titile_name
        # plots_err(
        #     [eval_steps1, eval_steps2],
        #     [y1_mean_scores, y2_mean_scores],
        #     [y1_std_scores, y2_std_scores],
        #     "Steps",
        #     "Scores",
        #     title,
        #     legends,
        # )

        # plots(
        #     [eval_steps1, eval_steps2],
        #     [y1_mean_scores, y2_mean_scores],
        #     "Steps",
        #     "Scores",
        #     title,
        #     legends,
        #     loc="upper left"
        # )

        # title = "Maximal Q-values in "+ titile_name
        # plots(
        #     [steps1, steps2],
        #     [y1q, y2q],
        #     "Steps",
        #     "Q values",
        #     title,
        #     legends,
        #     loc="upper left"
        # )



    # # ### Performance Profiles
    # perf_scores1 = perf_scores1/4900
    # perf_scores2 = perf_scores2/4900
    # #print(perf_scores1)
    # #print(perf_scores2)

    # title = "Performance Profiles"
    # plots(
    #     [perf_range, perf_range],
    #     [perf_scores1, perf_scores2],
    #     "Human Normalized Score (\u03C4)",
    #     "Fraction of Runs with Score > \u03C4",
    #     title,
    #     legends,
    #     loc="upper left"
    # )
