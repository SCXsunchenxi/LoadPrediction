import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import explained_variance_score
from alg.util import get_multi_linear_cols


'''''
岭回归RR模型
'''''

plt.rcParams['font.sans-serif']=['SimHei','FangSong','KaiTi','Arial Unicode MS']
plt.rcParams['axes.unicode_minus']=False
important_feature_number=20



'''''
RR模型类
'''''
class RidgeRegression(object):
    '''''
    RR初始化函数,保存模型的基本信息
    data_path: 模型的训练数据路径
    idx: 模型编号
    plane: 模型的飞机编号
    model_number: 多模型个数
    model_list: 多模型列表
    strain_number: 预测应变的个数
    multi_linear: 多重共线性
    save_dir: 模型保存路径
    feat_cols_path: 自变量特征
    target_col: 因变量特征
    '''''
    def __init__(self, model_name, idx, data_path,plane,model_number,multi_linear,strain_number,calibration_plane): # 添加了calibration_plane实现机动模型的标定--------------
        self.data_path = data_path
        self.idx = idx
        self.plane = plane
        self.model_number=model_number
        self.model_list=[]
        self.strain_number=strain_number
        self.multi_linear=multi_linear
        self.save_dir = os.path.join('static/model', model_name,str(idx),plane)
        os.makedirs(self.save_dir, exist_ok=True)
        self.model_save_path = os.path.join(self.save_dir, f'model.pkl')
        self.feat_cols_path = os.path.join(self.save_dir, f'feat_cols.pkl')
        if (strain_number!=0):
            if(model_number==0 or isinstance(model_number,list)): # 如果是机动划分或者混合划分----------------------
                listfile = os.listdir(self.data_path)
                for file in listfile:
                    if(file.startswith(self.plane)):
                        df = pd.read_csv(self.data_path + file)
                        self.target_col = df.columns.tolist()[-strain_number:]
                        break
            else:
                df = pd.read_csv(self.data_path + self.plane + '_data_' + str(0) + '.csv')
                self.target_col = df.columns.tolist()[-strain_number:]
        self.feat_cols = []
        self.data_exist=[] # 训练模型时是否有数据的标识---------------------------------
        self.data_exist_path = os.path.join(self.save_dir, f'data_exist.pkl') # 是否有数据标识保存路径---------------------------------

        self.calibration_plane=calibration_plane  # 添加了calibration_plane实现机动模型的标定--------------
    '''''
    RR的模型训练函数
    '''''
    def train(self):
        print('[*******************************************]RR MODEL TRAINING PROCESS')

        submodel_acc_list=[]# 每个子模型的单独准确度-----------------------------

        # 加载训练数据
        file_list=[]

        # 如果是机动多模型或混合多模型，需要按照机动编号读取数据
        if (self.model_number==0 or isinstance(self.model_number,list)): #-----------------
            create_model_number = 0
            listfile = os.listdir(self.data_path)
            for file in listfile:
                if file.startswith(str(self.plane+'_data')):
                    create_model_number=create_model_number+1
                    file_list.append(file)

        # 否则，按照模型个数编号读取数据
        else:
            create_model_number = self.model_number

        # 为每个子模型的每个应变初始化预测模型结构
        for m in range(create_model_number):
            models = []
            for i in range(self.strain_number):
                models.append(Ridge())
            self.model_list.append(models)

        results_accuracy=0 # 存储结果准确度
        r_number_model=create_model_number


        # 对每个子模型单独训练
        for m in range(create_model_number):

            # 保留无法训练的模型的准确度-------------------
            acc_ = 0
            r_number_ = self.strain_number

            # 获取机动数据、混合数据
            if (self.model_number == 0 or isinstance(self.model_number,list)): # -------------------
                file_path=os.path.join(self.data_path,file_list[m])
                df = pd.read_csv(file_path)

            # 获取天空点数据或整体数据
            else:
                df = pd.read_csv(self.data_path + self.plane + '_data_' + str(m) + '.csv')

            strain_number = self.strain_number
            if (len(df) == 0):
                feat_col = df.columns.tolist()[:-strain_number]
                self.feat_cols.append(feat_col)
                r_number_model = r_number_model - 1
                self.data_exist.append(0) # 训练此模型时无数据-------------------
                submodel_acc_list.append(0)  # 无法训练的子模型的准确度为0------------------
                continue

            self.data_exist.append(1)# 训练此模型时有数据-------------------
            # 去除数据的多重共线性
            feat_col = df.columns.tolist()[:-strain_number]
            # 无数据，则只保存去除多重共线性的数据，跳过不训练
            if not self.multi_linear:
                drop_cols = get_multi_linear_cols(df[feat_col])
                feat_col = df[feat_col].columns.difference(drop_cols).tolist()
            self.feat_cols.append(feat_col)
            X = df[self.feat_cols[m]]

            # 模型训练
            acc=0 # 去除无法训练的模型的准确度--------------------
            r_number = self.strain_number

            # 为每个应变单独训练一个预测模型
            for i, model in enumerate(self.model_list[m]):
                # 真实应变
                ys = df[self.target_col[i]]
                # 模型训练
                model.fit(X, ys)
                # 预测应变
                y_pre=model.predict(X)
                # 准确度
                r=explained_variance_score(y_pre, ys)
                print('[***] 应变 ' + str(i) + ' 准确度 ' + str(r))

                if(r>0.5):
                    acc=acc+r
                else: # 去除无法训练的模型
                    r_number=r_number-1
                acc_=acc_+r # 每个子模型的准确度------------------

            if (r_number > 0):
                acc = acc / r_number
                results_accuracy = results_accuracy + acc
                # 每个子模型的准确度------------------
                acc_=acc_/r_number_
                submodel_acc_list.append(acc_)
            else:
                r_number_model = r_number_model - 1
                submodel_acc_list.append(0)  # 每个子模型的准确度------------------

        # 返回训练模型的准确度
        if (r_number_model == 0):
            results_accuracy = -1
            print('[***] 数据量太小了！！！')
        else:
            results_accuracy = results_accuracy / r_number_model
        print('[***]Training process done')

        # 混合模型要返回创建模型的机动划分个数和天空点划分个数--------------------
        if (isinstance(self.model_number,list)):
            return results_accuracy, submodel_acc_list,[int(create_model_number/self.model_number[-1]),self.model_number[-1]]

        return results_accuracy,submodel_acc_list,create_model_number # submodel_acc_list------------------


    '''''
    RR的模型保存函数
    '''''
    def save_model(self):

        # 保存模型列表
        f= open(self.model_save_path, 'wb')
        pickle.dump(self.model_list, f)
        f.close()

        # 保存模型的因变量特征
        f=open(self.feat_cols_path, 'wb')
        pickle.dump(self.feat_cols, f)
        f.close()

        # 保存训练模型时是否有数据的列表-----------------------
        f = open(self.data_exist_path, 'wb')
        pickle.dump(self.data_exist, f)
        f.close()

        # 保存机动划分的机动标识
        if(self.model_number==0 ):
            JD_list = []
            listfile = os.listdir(self.data_path)
            for file in listfile:
                if file.startswith(str(self.plane+'_data')):
                    JD_list.append(file.split('_')[-1].split('.')[0]) # 去掉int--------------
            JD_list_path=os.path.join(self.save_dir,'JD_list.pkl')
            f = open(JD_list_path, 'wb')
            pickle.dump(JD_list, f)

        # 保存混合划分的标识 ----------------------------------------
        elif (isinstance(self.model_number,list)):
            index_list=[]
            listfile = os.listdir(self.data_path)
            for file in listfile:
                if file.startswith(str(self.plane + '_data')):
                    index_list.append(file.split('_')[-2]+'_'+(file.split('_')[-1]).split('.')[0])
            index_list_path = os.path.join(self.save_dir, 'index_list.pkl')
            f = open(index_list_path, 'wb')
            pickle.dump(index_list, f)

        print('[***]' + self.plane + ' model has saved!')


    '''''
    RR的模型加载函数
    '''''
    def load_model(self,pre_strain_number):

        # 加载模型列表
        f=open(self.model_save_path, 'rb')
        self.model_list = pickle.load(f)
        f.close()

        # 加载模型的自变量特征
        f=open(self.feat_cols_path, 'rb')
        self.feat_cols=pickle.load(f)
        f.close()

        # 加载训练模型时是否有数据的列表---------------------------
        f = open(self.data_exist_path, 'rb')
        self.data_exist = pickle.load(f)
        f.close()

        # 加载模型的预测因变量特征
        mean_std_path = os.path.join('static/model/线性回归(RR)', str(self.idx), self.plane, '0_mean_sd.csv')
        mean_std = pd.read_csv(mean_std_path, encoding='utf-8')
        self.target_col=mean_std.columns.tolist()[-pre_strain_number:]

        print('[***]' + self.plane + ' model has loaded!')


    '''''
    RR的模型预测方法，使用模型序列对每个应变预测
    '''''
    def predict(self, x,m):
        res = [model.predict(x).tolist() for model in self.model_list[m]]
        return res


    '''''
    根据模型的均值和方差还原预测数值
    '''''
    def restore_value(self, data,m):

        # 获取均值方差文件
        mean_std_path=os.path.join('static/model/线性回归(RR)', str(self.idx), self.plane, str(m) + '_mean_sd.csv')
        mean_std = pd.read_csv(mean_std_path, encoding='utf-8')
        # 均值
        pre_mean = mean_std.iloc[0:1,-self.strain_number:]
        # 方差
        pre_std = mean_std.iloc[1:2, -self.strain_number:]

        # 数据还原
        for col in self.target_col:
            data[col] = data[col] * float(pre_std[col]) + float(pre_mean[col])
        return data


    '''''
    根据模型的均值和方差处理预测时的飞参数据
    '''''
    def process_value(self, data, m):

        # 获取均值方差文件
        mean_std_path = os.path.join('static/model/线性回归(RR)', str(self.idx), self.plane, str(m) + '_mean_sd.csv')
        mean_std = pd.read_csv(mean_std_path, encoding='utf-8')
        if (self.strain_number != 0):
            feature_mean = mean_std.iloc[0:1, :-self.strain_number]
            feature_std = mean_std.iloc[1:2, :-self.strain_number]
        else:
            feature_mean = mean_std.iloc[0:1, :]
            feature_std = mean_std.iloc[1:2, :]

        # 还原数据
        for col in self.feat_cols[m]:
            data[col] = (data[col] - float(feature_mean[col])) / float(feature_std[col])
        return data


    '''''
    RR的模型测试/预测函数
    '''''
    def test(self, data_dir,record_idx):
        print('[*******************************************]RR MODEL TEST PROCESS')

        submodel_acc_list=[]# 每个子模型的单独准确度-----------------------------

        # 创建保存预测值的路径
        prediction_value_path = 'static/data/prediction/记录'+str(record_idx)
        os.makedirs(prediction_value_path, exist_ok=True)

        # 需要预测的飞机数据---------------------------------
        if(self.calibration_plane==None):
            plane = self.plane
        else:
            plane=self.calibration_plane

        print('[***]'+plane+' Predicting')
        y_df=pd.DataFrame() # 原值文件
        pre_df=pd.DataFrame() # 预测值文件


        # 机动多模型的预测方法
        if (self.model_number==0):

            # 读取机动标识列表
            JD_list_path = os.path.join(self.save_dir, 'JD_list.pkl')
            JD_list = pickle.load(open(JD_list_path, 'rb'))
            model_number=len(JD_list)

            # 使用子模型对对应的数据进行预测
            for i in range(model_number):

                m = JD_list[i]  # 当前模型的机动标识

                # 如果模型在训练时没有数据，就不使用此模型进行预测-------------------------
                if (self.data_exist[i]==0):
                    submodel_acc_list.append([m, 0])
                    continue

                # 读取对应机动标识的数据
                if (record_idx<0):
                    data_path=data_dir+'_data_'+str(m)+'.csv'
                else:
                    data_path=data_dir+plane+'_data_'+str(m)+'.csv'

                # 不存在此标识的数据，跳过
                if(not (os.path.exists(data_path))):
                    continue

                # 数据读取
                df = pd.read_csv(data_path)
                # 当前标识下的数据为空，跳过
                if(len(df)==0):
                    continue

                # 飞参特征数据
                X = df[self.feat_cols[i]]

                # 对飞参特征数据按均值方差处理
                X = self.process_value(X, i)

                # 预测值
                pred = self.predict(X, i)
                # 通过均值方差恢复原值
                pred_df = pd.DataFrame(np.array(pred).T, columns=self.target_col)
                # 加上索引编号，为了和原始值对齐
                index_df = df['origin_index']
                res_df = pd.concat([index_df, pred_df], axis=1)
                # 通过均值方差将预测值恢复
                res_df = self.restore_value(res_df, i)
                pre_df = pd.concat([pre_df, res_df], axis=0)

                # 若存在原始的值，则将预测和原始对齐
                if (self.strain_number != 0):
                    y = df[self.target_col]
                    y = pd.concat([index_df, y], axis=1)
                    y_df = pd.concat([y_df, y], axis=0)

                    # 计算每个模型的准确度并画图，标定过程不需要画图---------------------------------------
                    if(self.calibration_plane==None):
                        submodel_acc = 0
                        for s in self.target_col:
                            # 准确度
                            real_value=y[s].tolist()
                            predict_value=res_df[s].tolist()
                            submodel_acc = submodel_acc + explained_variance_score(real_value, predict_value)
                            # 画图
                            plt.title('飞机' + str(plane) + s + '的预测效果'+'(子模型'+str(m)+')')
                            plt.xlabel('真实值')
                            plt.ylabel('预测值')
                            plt.plot(real_value, real_value, '-', c='black', linewidth=1)
                            plt.plot(real_value, predict_value, 'o', c='red', markersize=2)
                            prediction_figure_file = os.path.join(prediction_value_path,
                                                                  '模型' + str(self.idx) + self.plane + s + '预测效果(子模型'+str(m)+').jpg')
                            plt.savefig(prediction_figure_file, dpi=300)
                            plt.close()
                        submodel_acc_list.append([m, submodel_acc / self.strain_number])


        # 混合划分多模型的预测方法 ----------------------------
        elif (isinstance(self.model_number,list)):

            # 读取混合划分标识列表
            index_list_path = os.path.join(self.save_dir, 'index_list.pkl')
            index_list = pickle.load(open(index_list_path, 'rb'))
            model_number = len(index_list)

            # 使用子模型对对应的数据进行预测
            for i in range(model_number):

                m = index_list[i]  # 当前模型的标识

                # 如果模型在训练时没有数据，就不使用此模型进行预测
                if (self.data_exist[i] == 0):
                    submodel_acc_list.append([m, 0])
                    continue

                # 读取对应机动标识的数据
                if (record_idx < 0):
                    data_path = data_dir + '_data_' + str(m) + '.csv'
                else:
                    data_path = data_dir + plane + '_data_' + str(m) + '.csv'
                # 不存在此标识的数据，跳过
                if (not (os.path.exists(data_path))):
                    continue
                # 数据读取
                df = pd.read_csv(data_path)
                # 当前标识下的数据为空，跳过
                if (len(df) == 0):
                    continue

                # 飞参特征数据
                X = df[self.feat_cols[i]]
                # 对飞参特征数据按均值方差处理
                X = self.process_value(X, i)

                # 预测值
                pred = self.predict(X, i)
                # 通过均值方差恢复原值
                pred_df = pd.DataFrame(np.array(pred).T, columns=self.target_col)
                # 加上索引编号，为了和原始值对齐
                index_df = df['origin_index']
                res_df = pd.concat([index_df, pred_df], axis=1)
                # 通过均值方差将预测值恢复
                res_df = self.restore_value(res_df, i)
                pre_df = pd.concat([pre_df, res_df], axis=0)

                # 若存在原始的值，则将预测和原始对齐
                if (self.strain_number != 0):
                    y = df[self.target_col]
                    y = pd.concat([index_df, y], axis=1)
                    y_df = pd.concat([y_df, y], axis=0)

                    # 计算每个模型的准确度并画图，标定过程不需要画图
                    if (self.calibration_plane == None):
                        submodel_acc = 0
                        for s in self.target_col:
                            # 准确度
                            real_value = y[s].tolist()
                            predict_value = res_df[s].tolist()
                            submodel_acc = submodel_acc + explained_variance_score(real_value, predict_value)
                            # 画图
                            plt.title('飞机' + str(plane) + s + '的预测效果' + '(子模型' + str(m) + ')')
                            plt.xlabel('真实值')
                            plt.ylabel('预测值')
                            plt.plot(real_value, real_value, '-', c='black', linewidth=1)
                            plt.plot(real_value, predict_value, 'o', c='red', markersize=2)
                            prediction_figure_file = os.path.join(prediction_value_path,
                                                                  '模型' + str(
                                                                      self.idx) + self.plane + s + '预测效果(子模型' + str(
                                                                      m) + ').jpg')
                            plt.savefig(prediction_figure_file, dpi=300)
                            plt.close()
                        submodel_acc_list.append([m, submodel_acc / self.strain_number])


        # 天空点多模型/整体模型的预测方法
        else:
            # 使用子模型对对应的数据进行预测
            for m in range(self.model_number):

                # 如果模型在训练时没有数据，就不使用此模型进行预测-------------------------
                if (self.data_exist[m] == 0):
                    submodel_acc_list.append([m, 0])
                    continue

                # 读取对应机动标识的数据
                if (record_idx<0):
                    data_path=data_dir+'_data_'+str(m)+'.csv'
                else:
                    data_path=data_dir+plane+'_data_'+str(m)+'.csv'
                df = pd.read_csv(data_path)
                # 当前类下的数据为空，跳过
                if (len(df) == 0):
                    continue

                # 飞参特征数据
                X = df[self.feat_cols[m]]
                # 对飞参特征数据按均值方差处理
                X=self.process_value(X,m)

                # 预测
                pred = self.predict(X,m)

                # 预测值
                pred_df = pd.DataFrame(np.array(pred).T, columns=self.target_col)
                # 加上索引编号，为了和原始值对齐
                index_df = df['origin_index']
                res_df=pd.concat([index_df,pred_df],axis=1)
                # 通过均值方差恢复原值
                res_df=self.restore_value(res_df,m)
                pre_df=pd.concat([pre_df,res_df],axis=0)

                # 若存在原始的值，则将预测和原始对齐
                if (self.strain_number!=0):
                    y = df[self.target_col]
                    y = pd.concat([index_df, y], axis=1)
                    y_df = pd.concat([y_df, y], axis=0)

                    # 计算每个模型的准确度并画图，标定过程不需要画图---------------------------------------
                    if (self.calibration_plane == None):
                        submodel_acc = 0
                        for s in self.target_col:
                            # 准确度
                            real_value = y[s].tolist()
                            predict_value = res_df[s].tolist()
                            submodel_acc = submodel_acc + explained_variance_score(real_value, predict_value)
                            # 画图
                            plt.title('飞机' + str(plane) + s + '的预测效果' + '(子模型' + str(m) + ')')
                            plt.xlabel('真实值')
                            plt.ylabel('预测值')
                            plt.plot(real_value, real_value, '-', c='black', linewidth=1)
                            plt.plot(real_value, predict_value, 'o', c='red', markersize=2)
                            prediction_figure_file = os.path.join(prediction_value_path,
                                                                  '模型' + str(
                                                                      self.idx) + self.plane + s + '预测效果(子模型' + str(
                                                                      m) + ').jpg')
                            plt.savefig(prediction_figure_file, dpi=300)
                            plt.close()
                        submodel_acc_list.append([m, submodel_acc / self.strain_number])


        # 如果预测数据的机动标识和模型的机动标识完全不匹配（没有预测结果），则返回-2-----------------------
        if(len(pre_df)==0):
            return 0, 0, -2

        # 合并类回到原始排序
        pre_df.sort_values(['origin_index'], inplace=True)
        pre_df=pre_df.set_index(["origin_index"])
        if (self.strain_number != 0):
            y_df.sort_values(['origin_index'], inplace=True)
            y_df=y_df.set_index(["origin_index"])

        # 保存结果
        original_data_path='static/data/raw_data/test/'+plane+'_data.csv'
        original_data=pd.read_csv(original_data_path, encoding='utf-8')
        if (self.strain_number != 0):
            target_col_map = {target: '原' + target for target in self.target_col}
            original_data.rename(columns=target_col_map,inplace=True)
        original_data=original_data.set_index(["origin_index"])
        original_pre_df=pd.concat([original_data,pre_df],axis=1)
        prediction_value_file = os.path.join(prediction_value_path, '模型' + str(self.idx) + self.plane + '预测结果.csv')
        original_pre_df.to_csv(prediction_value_file)

        # 画图 准确度
        if (self.strain_number != 0 and record_idx> -1):
            print('[***]Drawing')

            # 画图
            res_acc = {}
            for s in self.target_col:
                x = np.array(y_df[s].tolist())
                y = np.array(pre_df[s].tolist())
                plt.title('飞机' + str(plane) + s + '的预测效果')
                plt.xlabel('真实值')
                plt.ylabel('预测值')
                plt.plot(x, x, '-', c='black', linewidth=1)
                plt.plot(x, y, 'o', c='red', markersize=2)
                prediction_figure_file = os.path.join(prediction_value_path,
                                                      '模型' + str(self.idx) + self.plane + s + '预测效果(整体).jpg')
                plt.savefig(prediction_figure_file, dpi=300)
                plt.close()

                # 计算结果的精度
                acc = explained_variance_score(pre_df[s].tolist(),y_df[s].to_list())
                res_acc[s] = acc
            return res_acc, submodel_acc_list, prediction_value_path # 返回整体准确度，每个模型的准确度，预测数据保存路径-----------------------
        return 0,0, prediction_value_path # 返回整体准确度（没有原始应变无法计算准确度，返回0），每个模型的准确度（没有原始应变无法计算准确度，返回0），预测数据保存路径-----------------------


    '''''
    RR的模型解释函数
    '''''
    def explain(self):
        print('[*******************************************]RR MODEL EXPLAIN PROCESS')

        # 创建保存解释结果的路径
        explain_dir=self.save_dir+'/explanation'
        os.makedirs(explain_dir, exist_ok=True)

        # 对每个应变模型分别解释
        for s in range(len(self.target_col)):

            # 加载子模型个数
            if(self.model_number==0): # 机动模型
                JD_list_path = os.path.join(self.save_dir, 'JD_list.pkl')
                JD_list=pickle.load(open(JD_list_path, 'rb'))
                create_model_number=len(JD_list)

            # 混合模型的子模型个数------------------
            elif(isinstance(self.model_number,list)):
                index_list_path = os.path.join(self.save_dir, 'index_list.pkl')
                index_list = pickle.load(open(index_list_path, 'rb'))
                create_model_number = len(index_list)

            else:
                create_model_number=self.model_number

            imp_file = pd.DataFrame()# 解释保存的文件

            # 对每个子模型分别解释
            for m in range(create_model_number):

                # 如果模型在训练时没有数据，就跳过-------------------------
                if (self.data_exist[m] == 0):
                    continue

                # 加载模型
                model = self.model_list[m]

                # 计算模型的特征重要度
                imps = np.abs(np.abs(model[s].coef_))
                # 特征变量
                labels=self.feat_cols[m]

                # 保存特征和对应的重要度值
                imp_file_m = pd.DataFrame(
                    sorted(zip(labels, imps), key=lambda x: x[1], reverse=True),
                    columns=['特征', '重要度' + str(m)],)

                if (m==0):
                    imp_file=imp_file_m
                else:
                    imp_file=pd.merge(imp_file, imp_file_m, on='特征')

            # 保存解释文件文件
            values_list = imp_file.columns.tolist()[1:]
            imp_file['重要度值'] = imp_file[values_list].sum(axis=1)
            imp_file = imp_file.loc[:, ['特征', '重要度值']]
            imp_file.sort_values('重要度值', inplace=True, ascending=False)
            imp_file.to_csv(explain_dir+'/'+self.target_col[s]+'.csv', index=False, sep=',')

            # 保存解释图片
            plt.figure(figsize=(10, 6), dpi=250, linewidth=5)
            #plt.cla()
            fig, ax = plt.subplots(figsize=(10, 4))
            feature=imp_file['特征'].tolist()[:important_feature_number]
            value=imp_file['重要度值'].tolist()[:important_feature_number]
            ax.bar(feature, value)
            ax.set_ylabel('重要性', fontsize=15)
            ax.set_xlabel('特征', fontsize=15)
            ax.set_xticks(feature)
            ax.set_xticklabels(feature, rotation=90)
            plt.tight_layout()
            plt.savefig(explain_dir +'/'+self.target_col[s] + '.jpg', dpi=300)
            plt.clf()
            plt.close()
        print('[***]Explain process done')


# 测试
if __name__ == "__main__":
    for plane in ['P123', 'P126', 'P125', 'P127', 'P124']:
        model = RidgeRegression('线性回归(RR)', 1, 'static/data/processed_data/train/', plane, 4, 0, 11)

        # 训练
        model.train()
        model.save_model()

        # 预测
        model.load_model(11)
        res_acc, prediction_value_path=model.test('static/data/processed_data/test/',0)
        print(res_acc)

        # 解释
        model.load_model(11)
        model.explain()