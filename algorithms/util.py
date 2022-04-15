import numpy as np
import pickle

def get_multi_linear_cols(df, threshold=0.94):
    """
    类似 左鸭翼偏度 和 右鸭翼偏度 的特征很可能存在共线性，
    计算两两之间的协方差，
    大于某个值的时候，随机去除左右特征中的一个。
    """
    cols = df.columns.tolist()
    l_cols = [x for x in cols if '左' in x and x.replace('左', '右') in cols]
    r_cols = [x.replace('左', '右') for x in l_cols]
    drop_cols = []
    for i in range(len(l_cols)):
        # 非常负相关的特征，都保留
        # if np.corrcoef(df[l_cols[i]].values, df[r_cols[i]].values)[0][1] > threshold:
        # 非常负相关的特征，随机保留其中一个
        if (
            np.abs(np.corrcoef(df[l_cols[i]].values, df[r_cols[i]].values)[0][1])
            > threshold
        ):
            if np.random.randint(0, 2) == 0:
                drop_cols.append(l_cols[i])
            else:
                drop_cols.append(r_cols[i])
    return drop_cols


def pk_load(data_path):
    """
    加载对象函数

    反序列化对象，将data_path文件中的数据解析为一个python对象。

    参数
    -------
    data:
    data_path: str

    """
    return pickle.load(open(data_path, 'rb'))

