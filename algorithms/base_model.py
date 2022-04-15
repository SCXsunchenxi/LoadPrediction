from alg.RidgeRegression import RidgeRegression
from alg.LightGBM import LightGBM
from alg.MultilayerPerceptron import MultilayerPerceptron

''''
基础的机器学习模型定义方法
通过参数中选择的模型类型，返回对应的模型函数来创建模型
'''

model_list = [
    '线性回归(RR)',
    '决策树(LightGBM)',
    '神经网络(MLP)']


def define_model(model_name, idx, data_path,plane,model_number,multi_linear,strain_number,calibration_plane=None): # #定义模型时，加上标定飞机的属性------
    '''''
    model_name: 模型名称
    idx: 模型编号
    data_path:训练数据路径
    plane: 模型的飞机编号
    model_number: 多模型个数
    multi_linear: 多重共线性
    strain_number: 预测应变的个数
    '''''

    if '线性回归(RR)' in model_name:
        return RidgeRegression(model_name, idx, data_path,plane,model_number,multi_linear,strain_number,calibration_plane)

    elif '决策树(LightGBM)' in model_name:
        return LightGBM(model_name, idx, data_path,plane,model_number,multi_linear,strain_number,calibration_plane)

    elif '神经网络(MLP)' in model_name:
        return MultilayerPerceptron(model_name, idx, data_path,plane,model_number,multi_linear,strain_number,calibration_plane)

    else:
        raise NotImplementedError(f"{model_name} not implemented.")
