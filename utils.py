#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
根据movielens的README中提到的信息，下面简单记录下对数据的一些分析准备：

1）、user：UserID::Gender::Age::Occupation::Zip-code
用户的类别特征为：UserID、Gender、Age、Occupation、Zip-code
用户没有连续特征
2）、movie：MovieID::Title::Genres
movie有两个类别特征：MovieID、Genres，其中Genres是个多类别特征，即同一个电影可能有多个Genres。
movie有一个字符特征：Title，在本代码中暂且不用，后续可以考虑通过各种方式进行处理，转化成特征并利用。
3）、rating：UserID::MovieID::Rating::Timestamp
rating记录了每个用户对电影的评分，为了简单处理，我们把3分以下当成负样本，3分及以上作为正样本，然后用二分类的方法来做，评估指标使用AUC。

"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

embedding_num = {'uid': 6040, 'movieid': 3706, 'gender': 2, 'age': 7, 'occ': 21, 'zip_code': 3439, 'genres': 18}
embedding_dim = {'uid': 12, 'movieid': 12, 'gender': 12, 'age': 12, 'occ': 12, 'zip_code': 12, 'genres': 12}

def data_preprocess(user_file, movie_file, rating_file):
    """把movielens数据集按照rating进行正负样本整理，以及对应的用户和电影的特征."""
    # 首先处理用户特征
    uid_feature_dict = {}
    movieid_feature_dict = {}
    genre_set = set()
    with open(user_file) as f:
        user_data = f.readlines()
        for line in user_data:
            uid, gender, age, occ, zip_code = line.strip().split("::")
            uid_feature_dict[uid] = [gender, age, occ, zip_code]
    with open(movie_file, encoding='ISO-8859-1') as f:
        movie_data = f.readlines()
        for line in movie_data:
            movieid, _, genres = line.strip().split("::")
            movieid_feature_dict[movieid] = genres.split("|")
            for genre in movieid_feature_dict[movieid]:
                genre_set.add(genre)
    genre_dict = dict(zip(list(genre_set), range(len(genre_set))))
    colunms = ["uid", "movieid", "gender", "age", "occ", "zip_code", "genres", "label"]
    data = []
    with open(rating_file) as f:
        rating_data = f.readlines()
        for line in rating_data:
            uid, movieid, rating, _ = line.split("::")
            if int(rating) >= 3:
                label = 1
            else:
                label = 0
            tmp_genres = [genre_dict[i] for i in movieid_feature_dict[movieid]]
            tmp_data = [uid, movieid, uid_feature_dict[uid][0], uid_feature_dict[uid][1], uid_feature_dict[uid][2],
                        uid_feature_dict[uid][3], tmp_genres, label]
            data.append(tmp_data)
    data_df = pd.DataFrame(data, columns=colunms)
    for col in ["uid", "movieid", "gender", "age", "occ", "zip_code"]:
        tmp_label_encoder = LabelEncoder()
        data_df[col] = tmp_label_encoder.fit_transform(data_df[col])
    # data_df.to_csv('result.csv', index=False)
    return data_df


if __name__ == '__main__':
    user_file, movie_file, rating_file = "movielens/users.dat", "movielens/movies.dat", "movielens/ratings.dat"
    data_preprocess(user_file, movie_file, rating_file)
