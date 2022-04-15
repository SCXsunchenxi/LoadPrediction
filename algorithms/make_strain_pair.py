import os
import pandas as pd
pd.set_option('display.max_columns', None)
import torch as pt
from alg import base_model

'''''
通过间接模型法创建应变对
'''''

# cpu/gpu
device_str = "cuda"
device = pt.device(device_str if pt.cuda.is_available() else "cpu")
pair_data_output_dir = 'static/data/calibration/calibration_pair/'


'''''
创建应变对
'''''
def create_pair(data_path, plane1, plane2, model_name, idx, model_number, multi_linear, strain_number, strain_flag):
    '''''
    data_path: 训练数据路径
    plane1: 基础飞机
    plane2: 被标定飞机
    model_name: 基础预测模型名称
    idx: 预测模型编号
    model_number: 预测模型多模型个数
    multi_linear: 预测模型多重共线性
    strain_number: 应变个数
    strain_flag: 是否含有原始应变
    '''''

    # 创建保存结果的路径
    os.makedirs(pair_data_output_dir, exist_ok=True)

    print('[***]load model')
    pre_strain_number = strain_number
    if (strain_flag):
        strain_number = 0

    # 使用飞机2的模型对飞机1的数据预测
    model_plane2 = base_model.define_model(model_name, idx, data_path, plane2, model_number, multi_linear, strain_number,calibration_plane=plane1) #定义模型时，加上标定飞机的属性------
    # 飞机2的模型
    model_plane2.load_model(pre_strain_number)
    # 飞机1的数据
    plane1_data_path=data_path+plane1
    # 预测
    _,_,prediction_value_path_p1_p2=model_plane2.test(plane1_data_path, -1)

    # 使用飞机1的模型对飞机2的数据预测
    model_plane1 = base_model.define_model(model_name, idx, data_path, plane1, model_number, multi_linear, strain_number,calibration_plane=plane2) #定义模型时，加上标定飞机的属性------
    # 飞机1的模型
    model_plane1.load_model(pre_strain_number)
    #飞机2的数据
    plane2_data_path = data_path + plane2
    # 预测
    _,_, prediction_value_path_p2_p1 = model_plane1.test(plane2_data_path, -1)


    # 如果预测数据的机动标识和模型的机动标识完全不匹配（没有预测结果），则文件是空-----------------------
    if (prediction_value_path_p1_p2== -2):
        p1_p2_pair = pd.DataFrame()
        if (prediction_value_path_p2_p1==-2):
            print(str(plane1)+str(plane1)+'当前数据的划分标识和原模型完全不匹配！')
    else:
        p1_p2_file = os.path.join(prediction_value_path_p1_p2, '模型' + str(idx) + plane2 + '预测结果.csv')
        p1_p2_pair = pd.read_csv(p1_p2_file)

    if (prediction_value_path_p2_p1==-2):
        p2_p1_pair = pd.DataFrame()
    else:
        p2_p1_file = os.path.join(prediction_value_path_p1_p2, '模型' + str(idx) + plane1 + '预测结果.csv')
        p2_p1_pair = pd.read_csv(p2_p1_file)

    concat_p1_p2_file = pd.concat([p1_p2_pair, p2_p1_pair], axis=0)
    concat_p1_p2_file = concat_p1_p2_file.dropna()
    concat_p1_p2_file_path = os.path.join(pair_data_output_dir, plane1 + plane2 + '_pair.csv')
    concat_p1_p2_file.to_csv(concat_p1_p2_file_path, index=False, sep=',')

    return pair_data_output_dir



