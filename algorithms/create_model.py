import os
import pandas as pd
import shutil
import pickle
from alg import data_process
from alg import base_model

''''
创建"飞参-应变"预测模型算法
'''

''''
数据处理函数，调用data_process.py文件
'''
def process_data(data_dir,columnName,strain_number,multi_model,multi_model_index,train_test):
    '''''
    data_dir:数据路径
    columnName: 特征名称
    strain_number: 预测应变的个数
    multi_model: 多模型标识
    multi_model_index: 多模型划分规则
    train_test: 处理训练数据还是测试数据
    '''''

    print('[*******************************************]DATA PROCESS')

    # 获取飞机列表
    listfile = os.listdir(data_dir)
    plane_list = []
    for file in listfile:
        new_plane = file.split('_')[0]
        if new_plane.startswith('P'):
            if new_plane in plane_list:
                continue
            else:
                plane_list.append(new_plane)

    plane_list.sort()
    # 同一架飞机数据整合一起、天空点划分、数据扩充
    data_path, model_number,_ = data_process.dataprocess(data_dir, plane_list, columnName, strain_number,
                                                       multi_model,multi_model_index, train_test)
    return plane_list,data_path,model_number


''''
模型创建函数
通过base_model.py调用于选择模型类型对应的算法文件
来创建模型define_model()
训练模型train()
解释模型explain()
'''
def create_model(plane_list,model_name, idx, data_path,model_number,multi_linear,strain_number):
    '''''
    plane_list: 需要创建模型的飞机列表
    model_name: 模型名称
    idx: 模型编号
    data_path: 训练数据路径
    model_number: 多模型框架下子模型的个数
    multi_linear: 多重共线性标识
    strain_number: 预测应变的个数
    '''''

    print('[*******************************************]BUILD MODEL')
    accuracy_all=0
    create_model_number_list = []

    multimodel_acc_list=[] # multimodel_acc_list---------------------

    for plane in plane_list:
        model=base_model.define_model(model_name, idx, data_path,plane,model_number,multi_linear,strain_number)
        # 模型训练
        accuracy,submodel_acc_list,create_model_number=model.train()  # submodel_acc_list---------------------
        model.save_model()
        # 模型解释
        model.explain()
        accuracy_all=accuracy_all+accuracy
        # 机动、混合-----------------------------
        if (model_number == 0 or isinstance(model_number,list)):
            create_model_number_list.append(create_model_number)
        multimodel_acc_list.append(submodel_acc_list) # multimodel_acc_list---------------------
    return accuracy_all/len(plane_list), multimodel_acc_list, create_model_number_list # multimodel_acc_list---------------------


''''
保存模型的训练数据信息
保存训练数据的均值和方差，用于以后的数据预测
'''
def save_meansd(data_path,plane_list,model_number,model_path):
    '''''
    data_path: 数据路径
    plane_list: 飞机列表
    model_number: 多模型框架下子模型的个数
    model_path: 模型保存列表
    '''''

    if isinstance(model_number,int):
        for plane in plane_list:
            for m in range(model_number):
                mean_std_df = pd.read_csv(data_path + plane + '_mean_sd_' + str(m) + '.csv')
                # 保存模型数据的均值和方差
                save_path=os.path.join(model_path,plane, str(m) + '_mean_sd.csv')
                mean_std_df.to_csv(save_path, index=False, sep=',')
    else:
        if isinstance(model_number[0],int): # 机动划分-----------------------
            for p in range(len(plane_list)):
                JD_list_path=os.path.join(model_path,plane_list[p],'JD_list.pkl')
                JD_list=pickle.load(open(JD_list_path, 'rb'))
                for i in range(model_number[p]):
                    mean_std_df = pd.read_csv(data_path + plane_list[p] + '_mean_sd_' + str(JD_list[i]) + '.csv')
                    save_path = os.path.join(model_path, plane_list[p], str(i) + '_mean_sd.csv')
                    mean_std_df.to_csv(save_path, index=False, sep=',')
        else: # 混合划分-----------------------
            for p in range(len(plane_list)):
                index_list_path=os.path.join(model_path,plane_list[p],'index_list.pkl')
                index_list=pickle.load(open(index_list_path, 'rb'))
                for i in range(int(model_number[p][0]*model_number[p][1])):
                    mean_std_df = pd.read_csv(data_path + plane_list[p] + '_mean_sd_' + str(index_list[i]) + '.csv')
                    save_path = os.path.join(model_path, plane_list[p], str(i) + '_mean_sd.csv')
                    mean_std_df.to_csv(save_path, index=False, sep=',')



''''
清理存储
删除模型训练上传的数据和中间生成的过程数据文件
'''
def detele_process_data():
    print('[*******************************************]DELETE MEMORY DATA')
    if (os.path.exists('static/data/raw_data')):
        shutil.rmtree('static/data/raw_data')
    if (os.path.exists('static/data/processed_data')):
        shutil.rmtree('static/data/processed_data')




if __name__ == "__main__":

    '''
    基本参数设置
    '''

    # 训练数据路径
    train_data_dir ='static/data/raw_data/train/'

    # 数据特征名称
    columnName = ['全机重量', '马赫数', '气压高度', '攻角', '侧滑角', '动压', '法向过载', '侧向过载', '轴向过载', '俯仰角', '横滚角', '真航向角', '滚转速率',
                  '俯仰速率', '偏航速率', '滚转角加速度', '俯仰角加速度', '偏航角加速度', '左鸭翼偏度', '右鸭翼偏度', '左前襟偏度', '右前襟偏度', '左外副翼偏度', '右外副翼偏度',
                  '左内副翼偏度', '右内副翼偏度', '左方向舵偏度', '右方向舵偏度', '机翼剪力电桥1', '机翼剪力电桥2', '机翼剪力电桥4', '机翼弯矩电桥1', '机翼弯矩电桥3',
                  '机翼弯矩电桥6', '鸭翼剪力电桥', '鸭翼弯矩电桥1', '垂尾剪力电桥', '垂尾弯矩电桥1', '机身弯矩电桥1', 'None']

    # 需要预测的飞参个数
    strain_number=11

    # 多模型方法选择
    # 整体模型: MM=-1, index_list=[]; 机动划分 MM=1, index_list=[]; 天空点划分 MM=0, index_list = [['马赫数',0.85],['法向过载',3.0]]
    # MM=2 混合划分(机动+天空点): index_list = [['法向过载',2.0]] ------------------
    MM=1
    index_list = []

    # 基础模型选择: 线性回归(RR), 决策树(LightGBM), 神经网络(MLP)
    model_name='线性回归(RR)'

    # 模型编号
    model_id=100

    # 多重共线性: 去掉多重共线性0, 不去掉多重共线性1
    multi_linear=0


    '''
    创建一个预测模型的过程
    '''

    # 数据处理
    plane_list, data_path, model_number = process_data(train_data_dir, columnName, strain_number, MM,
                                                                    index_list, 'train')

    # 模型创建、训练、保存
    accuracy,multimodel_acc_list, create_model_number_list = create_model(plane_list, model_name, model_id, data_path,
                                                      model_number, multi_linear, strain_number) # multimodel_acc_list---------------------------

    # 保存模型对应数据的均值方差
    model_save_dir=os.path.join('static/model', model_name, str(model_id)) # 模型保存路径

    if (model_number == 0 or isinstance(model_number,list)): # 机动、混合--------------------
        save_meansd(data_path, plane_list, create_model_number_list, model_save_dir)

    else: # 整体、天空点
        save_meansd(data_path, plane_list, model_number,model_save_dir)


    # 只删除过程数据
    if (os.path.exists('static/data/processed_data')):
        shutil.rmtree('static/data/processed_data')

    print(multimodel_acc_list)
    print('[*******************************************]PROCESS DONE!')