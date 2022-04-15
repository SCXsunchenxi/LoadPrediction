import os
import shutil
from alg import data_process, make_strain_pair,find_k,DT_explanation

'''''
系数标定方法主文件
'''''

k_output_dir='static/model/系数标定/'
label_data_output_dir='static/data/calibration/calibration_cluster/'


'''''
数据处理，主要调用data_process.py中的数据处理方法
'''''
def process_data(data_dir,columnName,strain_number,multi_model,multi_model_index):
    '''''
    data_dir:数据路径
    columnName: 特征名称
    strain_number: 预测应变的个数
    multi_model: 多模型标识
    multi_model_index: 多模型划分规则
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

    # 同一架飞机数据整合一起、天空点划分、数据扩充，调用data_process.py中的数据处理方法
    data_path, model_number,_ = data_process.dataprocess(data_dir, plane_list, columnName, strain_number,
                                                       multi_model,multi_model_index, 'calibration')
    return plane_list,data_path,model_number


'''''
通过间接预测模型获取应变对数据
主要调用make_strain_pair.py中的方法
'''''
def make_pair_by_model(data_path,plane_list,model_name,idx,model_number,multi_linear,strain_number,strain_flag):
    '''''
    data_path: 训练数据路径
    plane_list: 需要标定的飞机列表
    model_name: 基础预测模型名称
    idx: 预测模型编号
    model_number: 预测模型多模型个数
    multi_linear: 预测模型多重共线性
    strain_number: 应变个数
    strain_flag: 是否含有原始应变
    '''''

    print('[*******************************************]MAKE PAIR')

    # 为每一对飞机获取应变对
    pair_data_output_dir=''
    for p1 in range(len(plane_list) - 1):
        for p2 in range(p1 + 1, len(plane_list)):
            # 调用make_strain_pair.py中的方法
            pair_data_output_dir=make_strain_pair.create_pair(data_path, plane_list[p1],plane_list[p2],model_name, idx, model_number, multi_linear, strain_number, strain_flag)


    return pair_data_output_dir


'''''
通过混合聚类方法进行系数标定
主要调用find_k.py中的方法
'''''
def get_k(pair_data_dir,strain_number,plane_list,calibration_id):
    '''''
        pair_data_dir: 应变对数据
        plane_list: 需要标定的飞机列表
        strain_number: 应变个数
        calibration_id: 标定记录编号
    '''''

    print('[*******************************************]GET K')

    # 创建结果保存路径
    k_result_dir=os.path.join(k_output_dir,str(calibration_id))
    os.makedirs(k_result_dir,exist_ok=True)

    # 为每一对飞机进行标定
    calibration_result=[]
    for p1 in range(len(plane_list) - 1):
        for p2 in range(p1 + 1, len(plane_list)):
            # 调用find_k.py中的方法
            result=find_k.calibrat_k(pair_data_dir,plane_list[p1],plane_list[p2],strain_number,k_result_dir)
            calibration_result.append(result)
    return calibration_result


'''''
对标定进行解释
DT_explanation.py中的方法
'''''
def explain_calibration(idx,method):
    print('[*******************************************]CALIBRATION EXPLAINING')

    # 创建结果保存路径
    listfile = os.listdir(label_data_output_dir)
    output_dir=os.path.join(k_output_dir,str(idx),'解释')
    os.makedirs(output_dir,exist_ok=True)

    # 为每一个标定进行
    for file in listfile:
        if(file=='.DS_Store'):
            continue
        file_path=os.path.join(label_data_output_dir,file)
        # DT_explanation.py中的方法
        DT_explanation.explain_cluster(file_path,method,output_dir)
    return label_data_output_dir,k_output_dir



'''''
删除中间产生的过程数据
'''''
def detele_process_data():
    print('[*******************************************]DELETE MEMORY DATA')
    if(os.path.exists('static/data/raw_data')):
        shutil.rmtree('static/data/raw_data')
    if (os.path.exists('static/data/processed_data')):
        shutil.rmtree('static/data/processed_data')
    if (os.path.exists('static/data/calibration/calibration_pair')):
        shutil.rmtree('static/data/calibration/calibration_pair')
    if (os.path.exists('static/data/calibration/calibration_cluster')):
        shutil.rmtree('static/data/calibration/calibration_cluster')
    if (os.path.exists('static/data/prediction/记录-1')): # 删除过程数据-----------------------------------------
        shutil.rmtree('static/data/prediction/记录-1')


'''''
文本展示
'''''
def show_txt(path):
    content = []
    for line in open(path, "r",encoding='utf-8'):
        content.append(line)
    return content

    

if __name__ == "__main__":

    '''
    基本参数设置
    '''

    # 需要预测的数据的路径 一定将数据放在此路径下
    calibration_data_dir ='static/data/raw_data/test/'

    # 数据特征名称
    columnName = ['全机重量', '马赫数', '气压高度', '攻角', '侧滑角', '动压', '法向过载', '侧向过载', '轴向过载', '俯仰角', '横滚角', '真航向角', '滚转速率',
                  '俯仰速率', '偏航速率', '滚转角加速度', '俯仰角加速度', '偏航角加速度', '左鸭翼偏度', '右鸭翼偏度', '左前襟偏度', '右前襟偏度', '左外副翼偏度', '右外副翼偏度',
                  '左内副翼偏度', '右内副翼偏度', '左方向舵偏度', '右方向舵偏度', '机翼剪力电桥1', '机翼剪力电桥2', '机翼剪力电桥4', '机翼弯矩电桥1', '机翼弯矩电桥3',
                  '机翼弯矩电桥6', '鸭翼剪力电桥', '鸭翼弯矩电桥1', '垂尾剪力电桥', '垂尾弯矩电桥1', '机身弯矩电桥1', 'None']

    # 需要标定的飞参个数
    strain_number=11

    # 使用的预测模型编号
    model_id=100

    # 使用的预测模型的多模型信息
    MM=1 # 整体模型: MM=-1, index_list=[]; 机动划分 MM=1, index_list=[]; 天空点划分 MM=0, index_list = [['马赫数',0.85],['法向过载',3.0]]
    index_list=[]
    model_name='线性回归(RR)' # 基础模型选择: 线性回归(RR), 决策树(LightGBM), 神经网络(MLP)
    multi_linear=0  # 多重共线性: 去掉多重共线性0, 不去掉多重共线性1

    # 标定记录的编号
    calibration_id=100



    '''
    应变系数标定
    '''

    # 数据处理
    plane_list, data_path, model_number = process_data(calibration_data_dir, columnName,
                                                                                 strain_number, MM, index_list)

    # 标定
    if (len(plane_list) < 2): # 系数标定至少有两架飞机的数据
        print('数据至少包含两架飞机')

    else:
        # 创建应变对, 返回应变对数据的路径
        pair_data_dir = make_pair_by_model(data_path, plane_list, model_name, model_id,
                                                                     model_number, multi_linear, strain_number, False)

        # 对应变对标定, 返回标定的结果
        result = get_k(pair_data_dir, strain_number, plane_list, calibration_id)

        # 标定解释, 返回解释路径
        label_data_dir, output_dir = explain_calibration(calibration_id, 'tree2b')

        # 删除所有过程数据，包括训练数据和中间数据
        # detele_process_data()
        # 只删除过程数据
        if (os.path.exists('static/data/processed_data')):
            shutil.rmtree('static/data/processed_data')
        if (os.path.exists('static/data/calibration/calibration_pair')):
            shutil.rmtree('static/data/calibration/calibration_pair')
        if (os.path.exists('static/data/calibration/calibration_cluster')):
            shutil.rmtree('static/data/calibration/calibration_cluster')

        print('[*******************************************]PROCESS DONE!')




















