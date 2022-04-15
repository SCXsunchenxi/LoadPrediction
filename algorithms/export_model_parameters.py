import torch as pt
import os
import joblib
import pickle

'''神经网络(MLP)
    信息:
        ***每个子模型的自变量特征***
        [[子模型0用到的自变量],
         [子模型1用到的自变量],
         ...
        ]
        ***模型参数***
        [子模型0参数
          [  
            第0-1层矩阵[矩阵行[],[]],
            第1-2层矩阵[矩阵行[],[]],
            ...
           ],
         子模型1参数[],
         ...
        ]
'''

''' 决策树(LightGBM)
    每个子模型的每个应变模型保存成一个单独的txt文件
    信息：
        整体森林的信息
        每个树的信息
        特征重要度
        超参数信息
'''


''' 线性回归(RR)
    信息：
        ***每个子模型的自变量特征***
        [[子模型0用到的自变量],
         [子模型1用到的自变量],
         ...
        ]
        
        ***每个子模型的回归系数***
        [
         [[子模型0应变0的回归系数],
          [子模型0应变1的回归系数],
         ...],
        [[子模型1应变0的回归系数],
         [子模型1应变1的回归系数],
         ...],
        ...
        ]
'''

# 需要导出的模型的信息
model_base_dir='../static/model/'
model_name='决策树(LightGBM)' # 模型类型: 线性回归(RR), 决策树(LightGBM), 神经网络(MLP)
model_id=0
plane='P125'
save_model_dir = os.path.join(model_base_dir,model_name,str(model_id),plane) # 模型所在的文件夹，如：../static/model/神经网络(MLP)/100/P123/


# 导出神网络模型MLP参数

if(model_name=='神经网络(MLP)'):
    PARA='' # 保存模型参数
    output_model_path = 'MLP_model_parameter.txt'  # 导出参数的路径

    # 遍历模型文件夹,找到所有子模型文件并按编号排序
    listfile = os.listdir(save_model_dir)
    model_file_list=[]
    for file in listfile:
        if file.endswith('.pt'):
            model_file_list.append(file.split('.')[0])
    model_file_list.sort()
    feature_file_path = os.path.join(save_model_dir, str('feat_cols.pkl'))
    feature_list = pickle.load(open(feature_file_path, 'rb'))  # 自变量特征列表

    PARA = '*******每个子模型的自变量特征*******\r\n'
    PARA = PARA + str(feature_list) + '\r\n'

    # 读取每个子模型文件，整合参数到整体模型中
    PARA = PARA + '*******每个子模型的参数*******\r\n'
    PARA_list=[]
    for model_file in model_file_list:
        model_file_path=os.path.join(save_model_dir,model_file+str('.pt'))
        model_parameters = pt.load(model_file_path)  # 加载模型
        parameter_list = []  # 参数保存列表
        # 获取矩阵的参数
        for parameter in model_parameters.parameters():
            parameter_list.append(parameter.tolist())
        PARA_list.append(parameter_list)
    PARA=PARA+str(PARA_list)

    # 将参数保存到txt文件
    if (os.path.exists(output_model_path)):# 删除同名文件
        os.remove(output_model_path)
    f = open(output_model_path, 'a') # 创建文件
    f.write(str(PARA)) # 保存参数
    print('[*****]MLP模型参数保存至' + output_model_path)


# 导出树模型LightGBM参数

elif(model_name=='决策树(LightGBM)'):
    output_model_dir = 'LightGBM_model_parameter'  # 导出参数的路径
    os.makedirs(output_model_dir,exist_ok=True)

    # 遍历模型文件夹,找到所有子模型文件并按编号排序
    listfile = os.listdir(save_model_dir)
    model_file_list=[]
    for file in listfile:
        if (file.endswith('pkl')):
            if (file=='feat_cols.pkl'):
                feature_file_path = os.path.join(save_model_dir, str('feat_cols.pkl'))
                feature_list = pickle.load(open(feature_file_path, 'rb'))  # 自变量特征列表
                feature_file_path = os.path.join(output_model_dir, str('feat_cols.txt'))
                f = open(feature_file_path, 'a')  # 创建自变量特征文件
                f.write(str(str(feature_list)))  # 保存
            else:
                model_file_list.append(file.split('.')[0])
    model_file_list.sort()

    # 读取每个子模型文件分别保存
    for model_file in model_file_list:
        model_file_path = os.path.join(save_model_dir, model_file + str('.pkl'))
        model=joblib.load(model_file_path)
        output_model_path=os.path.join(output_model_dir, model_file + str('.txt'))
        if (os.path.exists(output_model_path)):  # 删除同名文件
            os.remove(output_model_path)
        model.save_model(output_model_path) # 模型保存

    print('[*****]LightGBM模型参数保存至' + output_model_dir)

# 导出岭回归模型RR参数

else:
    output_model_path = 'RR_model_parameter.txt'  # 导出参数的路径

    # 加载模型
    model_file_path = os.path.join(save_model_dir,str('model.pkl'))
    model_list=pickle.load(open(model_file_path, 'rb')) # 模型列表
    feature_file_path = os.path.join(save_model_dir, str('feat_cols.pkl'))
    feature_list=pickle.load(open(feature_file_path, 'rb'))# 自 变量特征列表

    PARA = '*******每个子模型的自变量特征*******\r\n'
    PARA=PARA+ str(feature_list) +'\r\n'

    # 导出每个子模型的参数
    PARA =PARA+ '*******每个子模型的回归系数*******\r\n'
    PARA_list=[]
    for submodel in model_list:
        parameter_list=[]
        for submodel_singlestrain in submodel:
            parameter_list.append(submodel_singlestrain.coef_.tolist())
        PARA_list.append(parameter_list)
    PARA=PARA+str(PARA_list)

    # 将参数保存到txt文件
    if (os.path.exists(output_model_path)):# 删除同名文件
        os.remove(output_model_path)
    f = open(output_model_path, 'a') # 创建文件
    f.write(str(PARA)) # 保存参数

    print('[*****]RR模型参数保存至' + output_model_path)


