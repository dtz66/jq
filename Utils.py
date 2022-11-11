import pkuseg
import pandas as pd


train_data = pd.read_csv(
        'D:/桌面/BDCI-text-classification-master7/data/train.csv')
test_data = pd.read_csv(
        'D:/桌面/BDCI-text-classification-master7/data/test_new.csv')
seg = pkuseg.pkuseg()  # 以默认配置加载模型
train_data['segments'] = train_data['comment'].apply(lambda i: seg.cut(i))
test_data['segments'] = test_data['comment'].apply(lambda i: seg.cut(i))
train_data.to_csv(
        'D:/桌面/BDCI-text-classification-master7/data/segments/web_train.csv',
        index=False)
test_data.to_csv(
        'D:/桌面/BDCI-text-classification-master7/data/segments/web_test.csv',
        index=False)

