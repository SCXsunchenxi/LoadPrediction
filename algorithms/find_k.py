import pandas as pd
pd.set_option('display.max_columns', None)
import os
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

'''''
系数标定方法
'''''

plt.rcParams['font.sans-serif']=['SimHei','FangSong','KaiTi','Arial Unicode MS']
plt.rcParams['axes.unicode_minus']=False

cluster_min_samples = 2
draw_size = 0.1
label_data_output_dir='static/data/calibration/calibration_cluster/'


'''''
DBSCAN密度聚类
'''''
def clustering_DBSCAN(b_data,min_samples):
    print('[***]DBSCAN 开始')

    # 初始化参数
    # 轮廓系数记录
    max_clustring_score = -1
    # 根据数据规模和数据分布获得DBSCAN参数
    eps_default= (b_data.max() - b_data.min()) / 2000
    EPS=eps_default
    start_eps=eps_default/2
    if(start_eps<0.0001):
        start_eps=0.001
    print('[**]初始eps:' + str(start_eps))
    max_eps=start_eps*10
    step = start_eps / 10
    eps_mag=len(str(int(1 / step)))
    step=2 / (pow(10, eps_mag))
    range_eps = step*10
    number_of_cluster_max=1000
    final_labels=[]
    final_number_of_clusters=0

    print('[***]寻找最佳eps')
    # 根据轮廓系数迭代寻找用于聚类的初始eps参数
    print('[**]开始寻找初始eps')
    for eps in np.arange(start_eps, max_eps, step):
        db = DBSCAN(eps=eps).fit(b_data) # DBSCAN函数学习和拟合
        labels = db.labels_# 聚类标签
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) # 聚类个数
        print('[*]eps:'+str(eps)+';聚类个数:' + str(n_clusters_))
        if (n_clusters_ < number_of_cluster_max):
            start_eps=eps
            print('[**]初始eps确定为' + str(start_eps))
            break

    # 基于初始eps参数，根据轮廓系数迭代寻找最佳eps
    print('[**]开始寻找最佳eps')
    start_eps=start_eps+step
    end_eps=start_eps+range_eps
    for eps in np.arange(start_eps, end_eps, step):
        print('[*]eps:' + str(eps))
        db = DBSCAN(eps=eps,min_samples=min_samples).fit(b_data) # DBSCAN函数学习和拟合
        labels = db.labels_ # 聚类标签
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) # 聚类个数
        print('[*]聚类个数:' + str(n_clusters_))
        if(n_clusters_<2): # 聚类个数少于2则跳过
            continue
        clustring_score = metrics.silhouette_score(b_data, labels) # 轮廓系数得分
        print('[*]轮廓系数得分:' + str(clustring_score))
        # 保存最佳轮廓系数对应的DBSCAN的聚类参数
        if (clustring_score > max_clustring_score):
            max_clustring_score = clustring_score
            EPS = eps
            final_labels=labels
            final_number_of_clusters=n_clusters_

    print('[*****] DBSCAN 完成. 最佳eps:' + str(EPS)+ ';聚类个数'+str(final_number_of_clusters))
    return final_labels, final_number_of_clusters,EPS


'''''
根据正太分布确定初始的聚类
'''''
def clustering_NormalDistribution(b):

    # 根据b的精度来划分聚类的半径
    b = np.round(b, 2)
    b = np.concatenate([b, b], axis=1)
    # 为数据加上聚类的标识
    b_df = pd.DataFrame(b, columns=['b_value', 'label'])
    b_label_df = b_df.groupby('b_value').count().reset_index()
    # 为数据加上类别的标识
    for index, row in b_label_df.iterrows():
        b_value = row['b_value']
        b_df.label[b_df.loc[:, 'b_value'] == b_value] = index
    clusters_number=len(b_label_df)
    return b_df.values,clusters_number


'''''
标定的图片生成
'''''
def draw_calibration(strain_name, plane1,plane2, k,clusters_number,labeled_data,k_result_dir):
    '''''
    strain_name: 应变名称
    plane1: 基础飞机
    plane2: 标定飞机
    k: 标定系数
    clusters_number: b的聚类个数
    labeled_data: 聚类的标定数据
    k_result_dir: 标定系数结果路径
    '''''

    k_result_dir=os.path.join(k_result_dir,'图片')
    os.makedirs(k_result_dir,exist_ok=True)

    # 根据标定类别的个数来生成对应的颜色
    unique_labels = set(labeled_data['label'])
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    plt.title(strain_name + '标定K=' + str(k) + ', 共' + str(clusters_number) + '类')
    plt.xlabel('飞机' + str(plane1))
    plt.ylabel('飞机' + str(plane2))
    MIN = (labeled_data[['p1', 'p2']].min()).min()
    MAX = (labeled_data[['p1', 'p2']].max()).max()
    plt.xlim([MIN, MAX])
    plt.ylim([MIN, MAX])

    # 由于数据量大，为了展示更清洗，只画随机采样的点
    _, labeled_data = train_test_split(labeled_data, test_size=draw_size)
    for l, col in zip(unique_labels, colors):
        labeled_data_pair=labeled_data[labeled_data['label'] == l][['p1', 'p2']]
        if (len(labeled_data_pair)==0):
            continue
        one_clustering = np.array(labeled_data_pair)
        x, y = one_clustering[:, 0], one_clustering[:, 1]
        x_ = np.array([i for i in (range(int(MIN - 2), int(MAX + 2)))])
        y_ = x_ * k + (y - x * k).mean()
        plt.plot(x_, y_, '-', c=tuple(col), linewidth=0.5)
        plt.plot(x, y, 'o', c=tuple(col), markersize=2)

    plt.savefig(k_result_dir + '/'+plane1 + plane2 +'_' + strain_name + '标定.jpg',dpi=300)
    print('[*****]标定图片已保存')
    plt.close()


'''''
应变对系数标定的主方法
'''''
def calibrat_k(pair_data_dir,plane1,plane2,strain_number,k_result_dir):
    '''''
        pair_data_dir: 应变对数据
        plane1: 基础飞机
        plane2: 被标定飞机
        strain_number: 应变个数
        k_result_dir: 标定结果保存路径
    '''''

    # 创建结果保存路径
    os.makedirs(label_data_output_dir,exist_ok=True)
    os.makedirs(k_result_dir,exist_ok=True)

    # 应变对数据读取
    pair_data_file=os.path.join(pair_data_dir,plane1+plane2+'_pair.csv')

    # 不存在标定对，返回------------------------
    if (not (os.path.exists(pair_data_file))):
        result_plane = {}
        return result_plane


    pair_data=pd.read_csv(pair_data_file, encoding='utf-8')
    strain_name=pair_data.columns.tolist()[-strain_number:]
    #判断文件中是否有JD列
    JD_exist=0
    if(pair_data.columns.tolist()[-strain_number - 1]=='JD'):
        JD_exist=1
    calibration_file = pd.DataFrame()
    result_strain={} # 存储一个应变的标定结果
    result_plane={} # 存储一对飞机的所有应变的标定结果

    # 对每个应变进行系数标定
    for s in range(strain_number):
        print('[*******]标定飞机'+plane1 + '和' + plane2 + '的应变' + strain_name[s])

        # 应变数据,若有'JD'列则改变索引
        if(JD_exist):
            strain_index_x = strain_number * 2 - s + 1
        else:
            strain_index_x = strain_number*2 - s
        strain_index_y = strain_number - s
        x, y = np.array(pair_data)[:, -strain_index_x].reshape(-1, 1), np.array(pair_data)[:, -strain_index_y].reshape(-1, 1)

        # 初始化线性标定模型
        Linear_model = linear_model.LinearRegression()
        Linear_model.fit(x, y)

        # 计算初始k和参数
        k_initial = round(Linear_model.coef_[0][0], 2)
        y_pre = Linear_model.predict(x)
        r2_inital = metrics.r2_score(y, y_pre)
        step = 0.01
        range_k = 0.20
        r2_max = 0
        k_final = k_initial

        # 根据先验知识限制系数范围
        if (k_initial < 0.7 or k_initial>1.3):
            k_initial = 1.00
            range_k = 0.30
            step = 0.02
            if(np.random.randint(0,2)):
                range_k = 0.29

        print('[*****]K遍历搜寻...初始标定 K: ' + str(k_initial) + '; r2: ' + str(r2_inital))
        print('[***]K = 高斯分布标定开始')
        final_clusters_number=0 # 存储一个标定k下b的类数

        # 在一定范围内遍历k，最终选取拟合优度和聚类轮廓系数最好的k作为最终标定
        for k_candidate in np.arange(k_initial - range_k, k_initial + range_k, step):
            print('[*]K = ' + str(k_candidate) + ' 开始')

            # 候选k
            k_candidate = (np.round(k_candidate, 2))
            # 通过k计算数据的b
            b = y - x * k_candidate

            # 对b进行正太分布来聚类，调用clustering_NormalDistribution()方法，返回b对应的类别，类别个数
            labels, clusters_number = clustering_NormalDistribution(b)
            clustering_data = np.hstack((x,y, b, labels))
            clustering_data = pd.DataFrame(clustering_data, columns=['p1', 'p2', 'b', 'cluster_b', 'label'])

            # 在聚类结果下求新的拟合优度r2的值，是聚类后的每个b内部的r2的平均
            clustering_r2 = []
            for l in range(clusters_number):
                one_clustering = np.array(clustering_data[clustering_data['label'] == l][['p1', 'p2', 'b']])
                # 只有一个数据，r2=1
                if (len(one_clustering) == 1):
                    clustering_r2.append(1)
                    continue
                p1, p2 = one_clustering[:, 0], one_clustering[:, 1]
                b_mean = one_clustering[:, 2].mean()
                clustering_r2.append(metrics.r2_score(p2, p1 * k_candidate + b_mean))
            clustering_r2_mean = np.mean(clustering_r2)
            print('[*]r2: ' + str(clustering_r2_mean) + ' 聚类个数: ' + str(clusters_number))

            # 如果当前的拟合优度好，则保存此拟合优度下相关的k值和最终的聚类数
            if (r2_max < clustering_r2_mean):
                r2_max = clustering_r2_mean
                k_final = k_candidate
                final_clusters_number=clusters_number

        #  在正太分布下得到的标定结果
        b = y - x * k_final
        b = np.round(b, 2)
        b = np.concatenate([b, b], axis=1)
        b_df = pd.DataFrame(b, columns=['label', 'b_value'])
        b_label_df = b_df.groupby('b_value').count().reset_index()
        # 根据b对数据打上相应的类别标签
        for index, row in b_label_df.iterrows():
            b_value = row['b_value']
            b_df.label[b_df.loc[:, 'b_value'] == b_value] = index
        labeled_pair_data = np.hstack((b_df.values, x,y))
        labeled_pair_data = pd.DataFrame(labeled_pair_data, columns=['label', 'b_value', 'p1', 'p2'])

        feature_data = pair_data.iloc[:,1:-strain_number*2]
        labeled_data = pd.concat([feature_data, labeled_pair_data], axis=1)
        ori_labeled_data = labeled_data
        print('[*****]得到最终标定系数: ' + str(k_final))

        # 对结果进行密度合并（正太分布的两端的数据合并）
        print('[*****]对标定结果进行密度合并')
        cluster_number = labeled_data.groupby('label')['label'].count().tolist()
        # 寻找正太函数的两端的截断，两端有样本比较少的类别
        index_1 = 0
        for i in range(len(cluster_number)):
            if (cluster_number[i] > 10):
                index_1 = i
                break
        index_2 = 0
        for i in range(len(cluster_number) - 1, -1, -1):
            if (cluster_number[i] > 10):
                index_2 = i
                break
        # 获取两端类数量少的类别下的数据
        new_data_1 = labeled_data[labeled_data['label'] <= index_1]
        new_data_2 = labeled_data[labeled_data['label'] >= index_2]
        labeled_data = labeled_data[index_1 <= labeled_data['label']]
        labeled_data = labeled_data[labeled_data['label'] <= index_2]

        # 对两端样本少的类别进行基于密度聚类的合并
        # 对一端的样本少的类别进行基于密度聚类的合并
        x_1, y_1 = np.array(new_data_1['p1'].tolist()).reshape(-1, 1), np.array(new_data_1['p2'].tolist()).reshape(-1, 1)
        b_1 = y_1 - x_1 * k_final # 获取两端数据的b值
        # 调用clustering_DBSCAN对b进行聚类合并
        final_labels_1, final_number_of_clusters_1, _ = clustering_DBSCAN(b_1, cluster_min_samples)
        if(final_number_of_clusters_1!=0): # 合并成功
            # 将数据按照合并后的结果整合处理
            new_data_1['label'] = final_labels_1.reshape(-1, 1)
            new_data_1.drop(index=(new_data_1.loc[(new_data_1['label'] == -1)].index), inplace=True)
            for l in range(final_number_of_clusters_1):
                one_clustering = np.array(new_data_1[new_data_1['label'] == l][['p1', 'p2']])
                x, y = one_clustering[:, 0], one_clustering[:, 1]
                b_mean = (y - x * k_final).mean()
                new_data_1.loc[new_data_1[new_data_1['label'] == l].index, 'b_value'] = b_mean

            # 对另一端的样本少的类别进行基于密度聚类的合并
            x_2, y_2 = np.array(new_data_2['p1'].tolist()).reshape(-1, 1), np.array(new_data_2['p2'].tolist()).reshape(-1, 1)
            b_2 = y_2 - x_2 * k_final
            # 调用clustering_DBSCAN对b进行聚类合并
            final_labels_2, final_number_of_clusters_2, _ = clustering_DBSCAN(b_2, cluster_min_samples)
            if (final_number_of_clusters_2 != 0): # 合并成功
                # 将数据按照合并后的结果整合处理
                new_data_2['label'] = final_labels_2.reshape(-1, 1)
                new_data_2.drop(index=(new_data_2.loc[(new_data_2['label'] == -1)].index), inplace=True)
                for l in range(final_number_of_clusters_2):
                    one_clustering = np.array(new_data_2[new_data_2['label'] == l][['p1', 'p2']])
                    x, y = one_clustering[:, 0], one_clustering[:, 1]
                    b_mean = (y - x * k_final).mean()
                    new_data_2.loc[new_data_2[new_data_2['label'] == l].index, 'b_value'] = b_mean

                # 对两端合并后的新数据进行整合，重新编排类别标签
                labeled_data['label'] = labeled_data['label'] - index_1 - 1 + final_number_of_clusters_1
                new_data_2['label'] = new_data_2['label'] + final_number_of_clusters_1 + index_2 - index_1 - 1
                labeled_data = pd.concat([new_data_1, labeled_data, new_data_2], axis=0, ignore_index=True)
                # 计算合并后新的b的类别个数
                final_clusters_number = final_number_of_clusters_1 + index_2 - index_1 - 1 + final_number_of_clusters_2

                # 保存打标签数据
                labeled_data_path = os.path.join(label_data_output_dir,
                                                 plane1 + plane2 + strain_name[s] + '_cluster.csv')
                labeled_data.to_csv(labeled_data_path, index=False, sep=',')

            else: # 无法合并，则直接保存只按照正太分布聚类的结果
                # 保存打标签数据
                labeled_data_path = os.path.join(label_data_output_dir,
                                                 plane1 + plane2 + strain_name[s] + '_cluster.csv')
                ori_labeled_data.to_csv(labeled_data_path, index=False, sep=',')
                labeled_data=ori_labeled_data

        else: # 无法合并，则直接保存只按照正太分布聚类的结果
            # 保存打标签数据
            labeled_data_path=os.path.join(label_data_output_dir,plane1 +plane2 + strain_name[s]+ '_cluster.csv')
            ori_labeled_data.to_csv(labeled_data_path,index=False, sep=',')
            labeled_data = ori_labeled_data

        print('[*****]标定飞机' + plane1 + '和' + plane2 + '的' + strain_name[s] + '标定完成. k:' + str(k_final) + '. r2: ' + str(
            r2_max) + '. 聚类个数:' + str(final_clusters_number))

        print('[*****]聚类数据已保存')

        # 画pair的斜率图
        draw_calibration(strain_name[s], plane1, plane2, k_final, final_clusters_number, labeled_data,k_result_dir)

        # 标定信息整合
        b_value_df = pd.DataFrame(labeled_data.groupby('label').mean().reset_index()['b_value'].to_list(), columns=[strain_name[s] + '_k:' + str(k_final) + '_r2:' + str(r2_max) + '_类个数:' + str(clusters_number)])
        calibration_file = pd.concat([calibration_file, b_value_df], axis=1)
        result_strain[strain_name[s]]=[k_final,r2_max]

    # 保存标定结果
    plane_name=plane1+plane2
    result_plane[plane_name]=result_strain
    labeled_data_path = os.path.join(k_result_dir,plane1 + plane2 + '标定结果.csv')
    calibration_file.to_csv(labeled_data_path, index=False, sep=',')

    return result_plane