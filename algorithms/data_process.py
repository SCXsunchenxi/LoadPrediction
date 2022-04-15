import pandas as pd
pd.set_option('display.max_columns', None)
import os
import numpy as np
from sklearn.model_selection import train_test_split


''''
数据处理方法
'''

out_dir = 'static/data/processed_data/'
calibration_size=0.05

def dataprocess(data_dir,plane_list,columnName,strain_number,multi_model,multi_model_index,train_test):
    '''''
    data_dir:数据路径
    plane_list: 需要处理的飞机列表
    columnName: 特征名称
    strain_number: 预测应变的个数
    multi_model: 多模型标识
    multi_model_index: 多模型划分规则
    train_test: 处理训练数据还是测试数据
    '''''

    strain_flag=False  # 针对测试过程，标志测试数据是否含有原始的应变值
    listfile = os.listdir(data_dir)
    if(train_test=='train'):
        os.makedirs(out_dir + 'train/', exist_ok=True)
    # 针对测试过程，标志测试数据是否含有原始的应变值
    else:
        os.makedirs(out_dir + 'test/', exist_ok=True)
        # 判断是否有应变数据,忽略mac的DS_Store
        for file in listfile:
            if file.startswith('P'):
                if (pd.read_table(data_dir+file,sep=' ', encoding='utf-8').shape[1]==len(columnName)-strain_number):
                    columnName=columnName[:-(strain_number+1)]
                    columnName.append('None')
                    strain_flag=True
                    break

    # 对于每一架飞机进行单独的数据处理
    for plane in plane_list:

        # 数据整合: 将同一架飞机不同起落的数据整合在一起

        # 机动划分或混合划分(机动+天空点) -----------------
        if (multi_model==1 or multi_model==2): # -----------------
            DATA, M, S, number = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0
            for file in listfile:
                # 通过文件头的'Pxxx'标识来锁定当前飞机的数据
                if (file.startswith(plane) and not (file.endswith("_JD.dat"))):

                    # 如果当前文件没有对应的JD文件，那么就丢弃当前文件
                    JD_file = str(file.split('.')[0]) + '_JD.dat'
                    if (not os.path.exists(data_dir + JD_file)):
                        continue
                    data = pd.read_table(data_dir + file, sep=' ', encoding='utf-8', names=columnName) # 读取数据
                    # 错误文件
                    if (np.isnan(data.iloc[0, -2])):
                        continue
                    data = data.drop(['None'], axis=1) # 删除.dat文件最后一列多余的none列

                    # 通过文件的前两行还原成原始数值 **********************
                    for col in columnName:
                        if col not in ['None']:
                            mean = data[col][0] + 1000
                            std = data[col][1] + 1000
                            data[col] = data[col] * std + mean
                            data[col][0] = mean
                            data[col][1] = std
                    data = data.drop([0, 1]).reset_index(drop=True) # 丢掉前两行

                    JD_label=pd.read_table(data_dir + JD_file, sep=' ', encoding='utf-8', names=['label','JD','None']) # 读取JD数据
                    JD_label['JD']=JD_label['JD'].astype(int) # 将JD标签类型改成整型int
                    data=pd.concat([data, JD_label['JD']],axis=1)  # 为原数据添加JD标签列
                    DATA = pd.concat([DATA, data], ignore_index=True) # 将当前文件读取的数据添加到此飞机的整体数据上

                    # 没有上传机动标识文件则直接返回发出错误信息
                    if(len(DATA)==0):
                        print('[*****]请上传机动标识文件')
                        return out_dir, -2, strain_flag

        # 整体划分或天空点划分，此时不用考虑JD文件，因此和机动分类的数据整合处理方式不同
        else:
            DATA, M, S, number = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0
            for file in listfile:
                if (file.startswith(plane) and not (file.endswith("_JD.dat"))):
                    data = pd.read_table(data_dir + file, sep=' ', encoding='utf-8', names=columnName)
                    # 错误文件
                    if (np.isnan(data.iloc[0, -2])):
                        continue
                    data = data.drop(['None'], axis=1)

                    # 通过文件的前两行还原成原始数值 **********************
                    for col in columnName:
                        if col not in ['None']:
                            mean = data[col][0] + 1000
                            std = data[col][1] + 1000
                            data[col] = data[col] * std + mean
                            data[col][0] = mean
                            data[col][1] = std
                    data = data.drop([0, 1]).reset_index(drop=True)

                    DATA = pd.concat([DATA, data], ignore_index=True)


        # 如果当前的数据处理服务于系数标定，那么采样部分数据，为了提高后面聚类的效率
        if(train_test=='calibration'):
            sample = np.array(DATA)
            _, sample = train_test_split(sample, test_size=calibration_size)
            DATA = pd.DataFrame(sample, columns=DATA.columns.tolist())

        # 如果当前的数据处理服务于预测模型的测试或者标定模型的创建（即，不属于预测模型的训练），就需要在原始数据上加上索引，用于还原到之前的排列顺序
        if (train_test != 'train'):
            # 添加索引，为以后还原
            index_df = pd.DataFrame(range(DATA.shape[0]), columns=['origin_index'])
            DATA = pd.concat([index_df, DATA], axis=1)
            DATA.to_csv(data_dir+ plane + '_data.csv', index=False, sep=',')

        print(plane + ' integration done')


        # 机动划分: 将相同机动标识的数据整合在一起

        JD_index=[]
        if (multi_model==1):
            DF_list = []
            JD_index = DATA.groupby('JD').count().index.tolist() # 按照标识聚类
            # 为每一个标识提取数据，最后形成一个按标识的数据列表DF_list
            for jd_index in JD_index:
                one_JD_data=DATA.groupby('JD').get_group(jd_index)
                one_JD_data = one_JD_data.drop(['JD'], axis=1)
                DF_list.append(one_JD_data)

            print(plane + ' JD done')


        # 天空点划分: 将相同天空点分类的数据整合在一起

        elif (multi_model==0):
            DF_list = [DATA]
            df_list = []
            # 按照树杈划分的规则迭代划分，最后形成一个按天空点类的数据列表DF_list
            for index in multi_model_index:
                for DF in DF_list:
                    for i in range(len(index)):
                        if(i==0):
                            df=DF[DF[index[0]]<index[i+1]]
                        elif(i==len(index)-1):
                            df=DF[DF[index[0]]>index[i]]
                        else:
                            df = DF[(DF[index[0]] > index[i])&(DF[index[0]] < index[i+1])]
                        df_list.append(df)
                DF_list=df_list
                df_list=[]

            # 如果一个天空点划分内没有数据，则无法训练模型，需要改变划分方式
            if(train_test=='train'):
                for i in range(len(DF_list)):
                    if(len(DF_list[i])==0):
                        print("[*****] 天空点划分不合理，有无数据类别。")

            print(plane + ' sky point done')


        # 混合划分: 先机动划分再天空点划分-------------------------

        elif (multi_model==2):
            DF_list= []

            # 对数据进行机动划分
            DF_list_1= []
            JD_index = DATA.groupby('JD').count().index.tolist()  # 按照标识聚类
            # 为每一个标识提取数据，最后形成一个按标识的数据列表DF_list
            for jd_index in JD_index:
                one_JD_data = DATA.groupby('JD').get_group(jd_index)
                one_JD_data = one_JD_data.drop(['JD'], axis=1)
                DF_list_1.append(one_JD_data)

            # 对按照机动划分好的每个数据再进行天空点划分
            for one_DF in DF_list_1:
                DF_list_2 = [one_DF]
                df_list = []
                # 按照树杈划分的规则迭代划分，最后形成一个按天空点类的数据列表DF_list
                for index in multi_model_index:
                    for DF in DF_list_2:
                        for i in range(len(index)):
                            if (i == 0):
                                df = DF[DF[index[0]] < index[i + 1]]
                            elif (i == len(index) - 1):
                                df = DF[DF[index[0]] > index[i]]
                            else:
                                df = DF[(DF[index[0]] > index[i]) & (DF[index[0]] < index[i + 1])]
                            df_list.append(df)
                    DF_list_2 = df_list
                    df_list = []

                DF_list.extend(DF_list_2)

            # 每个数据集的机动标识，一个机动可在划分多个天空点
            SP_number = 1
            for index in multi_model_index: # 求出天空点划分个数
                SP_number = SP_number * len(index)
            JD_index_new=[]
            for JD in JD_index:
                for i in range(SP_number):
                    JD_index_new.append(str(int(JD))+'_'+str(i))
            JD_index=JD_index_new

            # 如果一个划分内没有数据，则无法训练模型，需要改变划分方式
            if (train_test == 'train'):
                for i in range(len(DF_list)):
                    if (len(DF_list[i]) == 0):
                        print("[*****] 天空点划分不合理，有无数据类别")

            print(plane + ' JDSP done')


        # 整体不划分

        else:
            DF_list=[DATA]


        # 扩充数据: 通过特征之间的乘积来扩充

        for i in range(len(DF_list)): # 对数据列表的每一类数据进行扩充
            DF_data=DF_list[i].copy(deep=True)
            fullcolname = DF_data.columns.tolist()
            M, S = pd.DataFrame(), pd.DataFrame()

            # 只对飞参数据扩充，不对预测的应变数据扩充
            if (strain_flag):
                DATA_extend=DF_data
            else:
                DATA_extend = DF_data.loc[:, fullcolname[:-strain_number]]
            colname = DATA_extend.columns.tolist()

            # 如果是测试数据，要从第2列开始扩充，因为第1列是之前加的为了还原的索引
            if (train_test == 'train'):
                column_begin = 0
            else:
                column_begin = 1

            # 乘积扩充
            for a in range(column_begin, len(colname) - 1):
                for b in range(a + 1, len(colname)):
                    DATA_extend[colname[a] + '*' + colname[b]] = DF_data[colname[a]] * DF_data[colname[b]]
            if (not strain_flag):
                DATA_extend[fullcolname[-strain_number:]] = DF_data.loc[:, fullcolname[-strain_number:]]


            # 标准化: 对最终的数据进行标准化处理，保存数据的均值和方差

            if (train_test == 'train'):
                for col in DATA_extend.columns.tolist():
                    if (col == 'origin_index'): # 索引列
                        M[col] = [0]
                        S[col] = [1]
                    else: # 数据列
                        mean = DATA_extend[col].mean()
                        std = DATA_extend[col].std()
                        M[col] = [mean]
                        S[col] = [std]
                        DATA_extend[col] = (DATA_extend[col] - mean) / std
                MS = pd.concat([M, S], ignore_index=True)

                DATA_extend=DATA_extend.dropna(axis=0, how='any')  # 去掉存在空值的行**********************************

                # 保存数据,当机动划分或混合划分时，数据按照机动编号的名称保存------------------------
                if (multi_model==1 ):
                    DATA_extend.to_csv(out_dir + 'train/' + plane + '_data_' + str(int(JD_index[i])) + '.csv', index=False, sep=',')
                    MS.to_csv(out_dir + 'train/' + plane + '_mean_sd_' + str(int(JD_index[i])) + '.csv', index=False, sep=',')
                elif (multi_model == 2):
                    DATA_extend.to_csv(out_dir + 'train/' + plane + '_data_' + str(JD_index[i]) + '.csv',
                                       index=False, sep=',')
                    MS.to_csv(out_dir + 'train/' + plane + '_mean_sd_' + str(JD_index[i]) + '.csv', index=False,
                              sep=',')
                else:
                    DATA_extend.to_csv(out_dir + 'train/' + plane + '_data_' + str(i) + '.csv', index=False, sep=',')
                    MS.to_csv(out_dir + 'train/' + plane + '_mean_sd_' + str(i) + '.csv', index=False, sep=',')

            else: # 模型测试时不需要将数据标准化，之后使用模型保存的均值和方差标准化,当机动划分或混合划分时，数据按照机动编号的名称保存------------------------

                DATA_extend=DATA_extend.dropna(axis=0, how='any')  # 去掉存在空值的行**********************************

                if (multi_model == 1 ):
                    DATA_extend.to_csv(out_dir + 'test/' + plane + '_data_' + str(int(JD_index[i])) + '.csv', index=False, sep=',')
                elif ( multi_model==2):
                    DATA_extend.to_csv(out_dir + 'test/' + plane + '_data_' + str(JD_index[i]) + '.csv', index=False, sep=',')
                else:
                    DATA_extend.to_csv(out_dir + 'test/' + plane + '_data_' + str(i) + '.csv', index=False, sep=',')

        print(plane + ' extend done')


    # 返回数据划分（多模型）的个数

    # 机动划分的模型个数不统一，不同的飞机的机动个数不同，返回0
    if (multi_model==1):
        model_number=0

    # 天空点划分的模型个数
    elif (multi_model==0):
        model_number=1
        for index in multi_model_index:
            model_number=model_number*len(index)

    # 混合划分的模型个数 --------------------
    elif (multi_model==2): #
        model_number = 1
        for index in multi_model_index:
            model_number = model_number * len(index)
        model_number= [0,model_number]

    # 整体模型的个数为1
    else:
        model_number = 1


    # 方法返回数据保存路径、模型个数、数据是否含有原始的应变
    if (train_test == 'train'):
        return out_dir+'train/', model_number,strain_flag

    else:
        return out_dir + 'test/', model_number,strain_flag


# 测试
if __name__ == "__main__":

    plane_list=['P123', 'P126', 'P125', 'P127', 'P124']
    strain_number=11
    multi_model_index=[['马赫数',0.85],['法向过载',3]]
    columnName=['全机重量', '马赫数', '气压高度', '攻角', '侧滑角', '动压', '法向过载', '侧向过载', '轴向过载', '俯仰角', '横滚角', '真航向角', '滚转速率', '俯仰速率', '偏航速率', '滚转角加速度', '俯仰角加速度', '偏航角加速度', '左鸭翼偏度', '右鸭翼偏度', '左前襟偏度', '右前襟偏度', '左外副翼偏度', '右外副翼偏度', '左内副翼偏度', '右内副翼偏度', '左方向舵偏度', '右方向舵偏度', '机翼剪力电桥1', '机翼剪力电桥2', '机翼剪力电桥4', '机翼弯矩电桥1', '机翼弯矩电桥3', '机翼弯矩电桥6', '鸭翼剪力电桥', '鸭翼弯矩电桥1', '垂尾剪力电桥', '垂尾弯矩电桥1', '机身弯矩电桥1', 'None']

    #out_dir,model_number=dataprocess('../../PLANE611/system/data/', plane_list, columnName,strain_number, multi_model_index, 'train')
    multi_model=0
    out_dir,model_number, strain_flag= dataprocess('static/data/raw_data/test/', plane_list, columnName, strain_number, multi_model,multi_model_index, 'test')



