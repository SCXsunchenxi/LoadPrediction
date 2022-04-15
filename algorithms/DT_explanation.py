import pandas as pd
import os
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from alg import tree2b

'''''
采用树模型对标定模型进行解释
'''''

plt.rcParams['font.sans-serif']=['SimHei','FangSong','KaiTi','Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
figure_out=0
max_depth=10


'''''
基本的树模型
'''''
def DT(x,y,b,output_path):

    # 调用决策树模型
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=2)

    # 通过"飞参-b"数据来训练分类器
    clf.fit(x, y)

    # 解释展示：可视化树的规则
    text_representation = tree.export_text(clf)
    for i, k in enumerate(x.columns.tolist()):
        text_representation = text_representation.replace(f'feature_{i}', k)
    for i in range(len(y)):
        text_representation = text_representation.replace(f'class: {y.values[i]}', f'类别{y.values[i]}'+': b='+str(b.values[i]))

    # 保存解释文本
    f=open(output_path,'w',encoding='utf-8')
    f.write(text_representation)


'''''
无重多叉树模型
'''''
def NonrepeatDT(file_path,output_dir):

    # 调用tree2b.py中的无重多叉树模型
    report, _, _ = tree2b.tree2text(file_path, second_dt_sample_num=100000, label_cnt_threshold=30, dt_max_depth=5)

    # 保存解释文本
    _, file_name = os.path.split(file_path)
    file_name = str(file_name.split('_')[0]) + '.txt'
    output_path = os.path.join(output_dir, file_name)
    f = open(output_path, 'w',encoding='utf-8')
    f.write(report)


'''''
整体调用方法
'''''
def explain_cluster(file_path,method,output_dir):

    # 使用普通决策树解释
    if(method=='DT'):
        data = pd.read_csv(file_path)
        data['label'] = data['label'].astype(int)
        x = data[data.columns.tolist()[:-4]]
        y = data['label']
        b = data['b_value']

        _, file_name = os.path.split(file_path)
        file_name = str(file_name.split('_')[0]) + '.txt'
        output_path = os.path.join(output_dir, file_name)
        DT(x, y, b,output_path)

    # 使用无重多叉树解释
    else:
        NonrepeatDT(file_path,output_dir)

    _, file_name = os.path.split(file_path)
    print('[*****]'+file_name+'解释完成')


'''''
以文本形式展示解释结果
'''''
def show_txt(plane1,plane2,s):

    calibration_output_dir = 'static/model/TreeExplanation_calibration/explanation/'
    file =  calibration_output_dir + 'tree_strain' + str(s) + '_' + plane1 + '_' + plane2 + '.txt'
    content = []
    for line in open(file, "r",encoding='utf-8'):
        content.append(line)
    return content


# 测试
if __name__ == "__main__":
    path = "../static/data/calibration/calibration_cluster/P125P127垂尾剪力电桥_cluster.csv"
    NonrepeatDT(path,'')
