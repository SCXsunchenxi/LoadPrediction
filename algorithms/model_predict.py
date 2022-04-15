import os
import pickle
import shutil
from alg import data_process
from alg import base_model


'''''
对飞参数据使用已创建的预测模型进行应变预测
'''''


'''''
测试数据处理，主要调用data_process.py的数据处理方法
'''''
def process_data(data_dir,columnName,strain_number,multi_model,multi_model_index,train_test):
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

    # 同一架飞机数据整合一起、天空点划分、数据扩充，调用data_process.py的数据处理方法
    data_path, model_number,strain_flag= data_process.dataprocess(data_dir, plane_list, columnName, strain_number,
                                                       multi_model,multi_model_index, train_test)
    return plane_list,data_path,model_number,strain_flag


'''''
预测
'''''
def predict(data_path,plane_list,model_name, idx,model_number,multi_linear,strain_number,strain_flag,record_idx):
    '''''
    data_path: 测试数据路径
    plane_list: 测试飞机列表
    model_name: 基础预测模型名称
    idx: 预测模型编号
    model_number: 预测模型多模型个数
    multi_linear: 预测模型多重共线性
    strain_number: 应变个数
    strain_flag: 是否含有原始应变
    record_idx: 预测记录编号
    '''''
    print('[*******************************************]BUILD MODEL')

    pre_strain_number=strain_number
    if (strain_flag):
        strain_number=0
    mean_accuracy=0
    res_acc=0  # 单个预测准确度
    acc_file={} # 储存所有的预测准确度
    prediction_value_path=''

    all_submodel_acc_list=[] # 每个飞机的每个子模型的准确度-----------------------------

    # 对每架飞机进行预测
    for plane in plane_list:

        # 对已创建的模型进行加载
        model=base_model.define_model(model_name, idx, data_path,plane,model_number,multi_linear,strain_number)
        model.load_model(pre_strain_number)
        # 预测，调用对应模型的test()函数
        res_acc, submodel_acc_list,prediction_value_path=model.test(data_path,record_idx) # 返回submodel_acc_list----------------

        if (prediction_value_path == -2):
            print(str(plane)+'当前数据的划分标识和原模型完全不匹配！')
            continue

        # 获取所有飞机的准确度，计算平均准确度 res_acc=0说明没有原始的应变值
        if (res_acc!=0):

            all_submodel_acc_list.append(submodel_acc_list)  # 每个飞机的每个子模型的准确度-----------------------------

            acc_file[plane] = res_acc
            one_mean_accuracy=0
            one_mean_accuracy_number=len(res_acc)
            for key in res_acc:
                if(res_acc[key]>0.5):
                    one_mean_accuracy=one_mean_accuracy+res_acc[key]
                else:
                    one_mean_accuracy_number=one_mean_accuracy_number-1

            if(one_mean_accuracy_number==0):
                mean_accuracy=0.5
            else:
                mean_accuracy=mean_accuracy+one_mean_accuracy/one_mean_accuracy_number

    # 保存预测准确度的信息，res_acc=0说明没有原始的应变值，不保存
    if (res_acc != 0):
        mean_accuracy=mean_accuracy/len(plane_list)

        acc_file_path = os.path.join(prediction_value_path, '模型' + str(idx) + '预测准确度_应变.pkl')
        f = open(acc_file_path, 'wb')
        pickle.dump(acc_file, f)
        f.close()


        # 保存每个飞机的每个子模型的准确度，保存到文件--------------------------
        all_submodel_acc_path = os.path.join(prediction_value_path, '模型' + str(idx) + '预测准确度_子模型.pkl')
        f = open(all_submodel_acc_path, 'wb')
        pickle.dump(all_submodel_acc_list, f)
        f.close()



    return mean_accuracy,prediction_value_path





if __name__ == "__main__":

    '''
    基本参数设置
    '''

    # 需要预测的数据的路径 一定将数据放在此路径下
    test_data_dir ='static/data/raw_data/test/'

    # 数据特征名称
    columnName = ['全机重量', '马赫数', '气压高度', '攻角', '侧滑角', '动压', '法向过载', '侧向过载', '轴向过载', '俯仰角', '横滚角', '真航向角', '滚转速率',
                  '俯仰速率', '偏航速率', '滚转角加速度', '俯仰角加速度', '偏航角加速度', '左鸭翼偏度', '右鸭翼偏度', '左前襟偏度', '右前襟偏度', '左外副翼偏度', '右外副翼偏度',
                  '左内副翼偏度', '右内副翼偏度', '左方向舵偏度', '右方向舵偏度', '机翼剪力电桥1', '机翼剪力电桥2', '机翼剪力电桥4', '机翼弯矩电桥1', '机翼弯矩电桥3',
                  '机翼弯矩电桥6', '鸭翼剪力电桥', '鸭翼弯矩电桥1', '垂尾剪力电桥', '垂尾弯矩电桥1', '机身弯矩电桥1', 'None']

    # 需要预测的飞参个数
    strain_number=11

    # 使用的模型编号
    model_id=100

    # 使用的模型的多模型信息
    # 整体模型: MM=-1, index_list=[]; 机动划分 MM=1, index_list=[]; 天空点划分 MM=0, index_list = [['马赫数',0.85],['法向过载',3.0]]
    # MM=2 混合划分(机动+天空点): index_list = [['法向过载',3.0]] ------------------
    MM = 2
    index_list = [['法向过载', 2.0]]


    model_name='神经网络(MLP)' # 基础模型选择: 线性回归(RR), 决策树(LightGBM), 神经网络(MLP)
    multi_linear=0  # 多重共线性: 去掉多重共线性0, 不去掉多重共线性1

    # 预测记录的编号
    record_id=100



    '''
    通过已有模型对数据预测
    '''

    # 数据处理
    plane_list, data_path, model_number, strain_flag = process_data(test_data_dir, columnName,strain_number,MM, index_list, 'test')

    # 使用模型对数据预测 all_submodel_acc_list--------------
    mean_accuracy, prediction_value_path = predict(data_path, plane_list, model_name, model_id, model_number,multi_linear, strain_number, strain_flag, record_id)


    # 只删除过程数据
    if (os.path.exists('static/data/processed_data')):
        shutil.rmtree('static/data/processed_data')

    print('[*******************************************]PROCESS DONE!')