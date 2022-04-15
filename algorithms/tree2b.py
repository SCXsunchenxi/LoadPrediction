
import pandas as pd
import numpy as np
from scipy import stats
from pprint import pprint
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from alg.chefboost import Chefboost as chef

"""
无重多叉树实现
"""


def tree2text(path, second_dt_sample_num=100000, label_cnt_threshold=10, dt_max_depth=5, verbose=False):
    """
        second_dt_sample_num: 训练C4.5的样本数量
        label_cnt_threshold: 去除样本数量小的类别
        dt_max_depth: CART最大深度，深度越大，获取的特征和分界线越多
    """
    # 加载数据
    df = pd.read_csv(path).drop(columns=['b_value','p1','p2'])
    df['label'] = df['label'].astype(int)
    x_col = list(df.columns)[:-1]
    y_col = 'label'

    # 忽略数据量少的类别
    dfvc = df['label'].value_counts()
    label_cnt_threshold=dfvc.max()/2
    ignore_label = (dfvc[dfvc < label_cnt_threshold]).index.tolist()
    print(f'ignore {len(ignore_label)} classes')
    df = df[~df.label.isin(ignore_label)]
    
    # 训练CART树
    print('training CART')
    clf = DecisionTreeClassifier(max_depth=dt_max_depth) # min_samples_leaf
    clf.fit(df[x_col], df[y_col])
    text_representation = tree.export_text(clf)
    
    # 根据训练得到的CART树，选择特征和切分点
    splits = {}
    def get_split_point(clf, node, splits):
        right_child_fmt = "{} <= {}"
        left_child_fmt = "{} >  {}"
        truncation_fmt = "{}"
        tree_ = clf.tree_
        # 特征改名
        feature_names_ = ["feature_{}".format(i) for i in clf.tree_.feature]
        # 得到树的分叉树的特征
        if tree_.feature[node] != tree._tree.TREE_UNDEFINED:
            name = feature_names_[node]
            threshold = tree_.threshold[node]
            if name not in splits.keys():
                splits[name] = set()
            splits[name].add(threshold)
            # 递归循环树获取分叉
            get_split_point(clf, tree_.children_left[node], splits)
            get_split_point(clf, tree_.children_right[node], splits)

    # 运行获取且分店的函数
    get_split_point(clf, 0, splits)
    # 如果数据量太少，无法解释
    if(len(splits)==0):
        report = "由于每类下数据太少,标定类别无法解释."
        return report, text_representation, 0
    print(f'{len(splits)} feats in total')

    # 根据新的特征和切分点，得到新的划分结果数据
    feat_col_idx = [int(x.split('_')[-1]) for x in list(splits.keys())]
    col_idx2name = {i:j for i, j in enumerate(x_col)}
    feat_col_name = [col_idx2name[i] for i in feat_col_idx]
    feat_col_name.append('label')
    df = df[feat_col_name]
    new_col_name = [x_col[int(x.split('_')[-1])] for x in splits.keys()]
    new_split = {col_idx2name[int(k.split('_')[-1])]: v for k, v in splits.items()}
    
    # 获取划分特征的字典序列
    split_name_dict = {}# split_name_dict 的 key 为特征名字（中文），value为切分点的集合（float）
    for feat, points in new_split.items():
        points = sorted(list(points))
        tmp_list = []
        for i in range(len(points)+1):
            if i == 0:
                tmp_list.append('feature <  {:.2f}'.format(points[i]))
            elif i == len(points):
                tmp_list.append('feature >= {:.2f}'.format(points[i-1]))
            else:
                tmp_list.append('{:.2f} <= feature < {:.2f}'.format(points[i-1], points[i]))
        split_name_dict[feat] = tmp_list
    if verbose:
        pprint(split_name_dict)

    # 对整体数据使用新的划分规则分割数据
    for i, x in enumerate(feat_col_idx):
        if verbose:
            print(x, col_idx2name[x])
        bins = sorted(list(splits[f'feature_{x}']))
        bins.insert(0, df[col_idx2name[x]].min()-1)
        bins.append(df[col_idx2name[x]].max()+1)
        classes = [str(i) for i in range(len(bins)-1)]
        df[col_idx2name[x]] = pd.cut(df[col_idx2name[x]], bins, labels=classes)
    df = df.rename(columns={'label': 'Decision'}).astype('object')
    df['Decision'] = df['Decision'].apply(lambda x: 'C' + str(x))
    
    # 训练 C4.5树
    # *c4.5 do not support max_depth. source code modified in chefboost/training/Training.py.
    # *parallelism went wrong, dont know why
    # *took ~1min to train on 100000 samples
    # *修改了代码，不能用 pip install chefboost的原始版本。改动包括：调整子节点顺序，记录并返回树的根节点等
    config = {'algorithm': 'C4.5', 'max_depth': 4, 'epochs': 5, 'enableParallelism': False}
    # model 包含决策树提取出来的规则（outputs/rules/rules/py/），可以应用于新的数据
    # 以及树的根节点 model['head']
    # 树中的每个节点包含 feature，next（子节点）， split（子节点对应的切分的名字）
    model = chef.fit(df[:second_dt_sample_num], config=config, do_eval=False)

    
    # 修改树的节点名字
    def modify_tree(r):
        if r.feature == 'Class':
            return
        new_splits_name = []
        for x in r.splits:
            new_splits_name.append(split_name_dict[r.feature][int(x)])
        r.splits = new_splits_name
        for sub in r.next:
            modify_tree(sub)
    modify_tree(model['head'])
    
    # 可视化，类似 sklearn 的 tree.export_text
    spacing=3
    child_fmt = "{} {}\n"
    truncation_fmt = "{} {}\n"
    value_fmt = "{}{}{}\n"
    report = ""
    def print_tree_recurse(node, depth):  # 可视化函数
        indent = ("|" + (" " * spacing)) * depth
        indent = indent[:-spacing] + "-" * spacing
        nonlocal report
        
        if node.feature == 'Class':
            val = ' class: ' + str(node.cl)
            report += value_fmt.format(indent, '', val)
        else:
            name = node.feature
            for i, subnode in enumerate(node.next):
                threshold = node.splits[i].replace('feature', node.feature)
                report += child_fmt.format(indent, threshold)
                print_tree_recurse(subnode, depth + 1)

    # 保存解释结果
    print_tree_recurse(model['head'], 1)
    df2 = pd.read_csv(path)
    data_b = df2[['label','b_value']]
    data_b = data_b.drop_duplicates()
    data_b_list=data_b.values.tolist()
    data_b_list.sort(key=lambda x :x[0]) #reverse=True
    data_b_list_back = data_b_list[::-1]
    for i in range(len(data_b_list_back)):
        report = report.replace(f'class: C{str(int(data_b_list_back[i][0]))}', f'b_value: {str(data_b_list_back[i][1])}')
    return report, text_representation, model


# 测试
if __name__ == "__main__":
    path = "P125P127机翼弯矩电桥6_cluster.csv"
    report, old_report, model = tree2text(path, second_dt_sample_num=100000, label_cnt_threshold=50, dt_max_depth=5)
    f = open('P123P124.txt', 'w')
    f.write(report)