#encoding=utf8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 读取原始数据
df = pd.read_csv("heart.csv")

# 查看前5行
print('-'*100)
print(df.head())
print('-'*100)

# 查看数据量
print('-'*100)
print(df.values.shape)
print('-'*100)
# 303个样本，14个字段

# 检查 各字段有没有缺失值
print('-'*100)
print(df.isnull().sum())
print('-'*100)
# 没有缺失值

data = df




# 散点图

# 箱型图
sns.swarmplot(x='exang',y='thalach',hue='target',data=data, size=6)
plt.xlabel('exang')
plt.ylabel('thalach')
plt.show()


# 查看target的分布情况
print('-'*100)
print(df.target.value_counts())
countNoDisease = len(df[df.target == 0])
countHaveDisease = len(df[df.target == 1])
print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))
print('-'*100)
sns.countplot(x="target", data=df, palette="bwr")
plt.show()
# 从图中看起来target的分布还是比较均衡的。
# 可以在此基础上进行模型的训练 和预测。

# 观察 年龄分布
sns.distplot(df['age'], color = 'cyan')
plt.show()

# 观察性别分布
size = df['sex'].value_counts()
colors = ['lightblue', 'lightgreen']
labels = "Male", "Female"
explode = [0, 0.01]
#(0,0)为圆心,0.7为半径
my_circle = plt.Circle((0, 0), 0.7, color = 'white') 
#绘制一个饼状图,参数explode为离开圆心的距离，autopct数据标签保留小数点后两位
plt.pie(size, colors = colors, labels = labels, shadow = True, explode = explode, autopct = '%.2f%%') 
plt.title('Distribution of Gender', fontsize = 20)
#获取当前图表
p = plt.gcf()
#p.gca()获取子图，add_artisi()将画好的饼状图添加进去
p.gca().add_artist(my_circle) 
plt.legend()
plt.show()

# 各性别患病
pd.crosstab(df.sex, df.target).plot(kind="bar", color=['#30A9DE','#EFDC05' ])
plt.title('sex - target')
plt.xlabel('sex (0 = Female, 1 = male)')
plt.xticks(rotation=0)
plt.legend(["no disease", "disease"])
plt.show()


# 各年龄患病
pd.crosstab(df.age, df.target).plot(kind="bar")
plt.title('age - target')
plt.xlabel('age')
plt.show()



# 'cp', 'thal' and 'slope' 三个字段是 离散型的数据， 也就是类别型的数据，
# 使用 pd.get_dummies()方法分别对这三个字段进行 one-hot encoding. 得到若干个新字段
a = pd.get_dummies(df['cp'], prefix = "cp")
print(a.info())
b = pd.get_dummies(df['thal'], prefix = "thal")
print(b.info())
c = pd.get_dummies(df['slope'], prefix = "slope")
print(c.info())
# 把新字段拼接到原始数据里面
frames = [df, a, b, c]
df = pd.concat(frames, axis = 1)
print('-'*100)
print(df.head())
print('-'*100)



# one-hot 编码之后，原始的3个字段就没用了，删除它
df = df.drop(columns = ['cp', 'thal', 'slope'])

# 看一下相关性
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,fmt='.1f')
plt.show()

# 删除相关性为0的字段
df = df.drop(columns = ['trestbps', 'chol',  'fbs', 'restecg', 'cp_3', 'thal_0', 'thal_1', 'slope_0'])

# 从数据集中把 特征x 和 target(也就是y)分离出来.  方便后面进行模型的训练
y = df.target.values
x_data = df.drop(['target'], axis = 1)

# 数据标准化 Normalize
# newX = (X - Xmin) / (Xmax - Xmin)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

# 对数据集进行分隔
# 80% 作为训练集数据
# 20% 作为测试集数据
# 在训练集数据上面进行模型的训练， 然后在测试集数据上进行模型的预测，并得到模型的各种评估指标。
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

print('-'*100)
print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))
print('-'*100)
# 从中可以看到，训练集样本数为242， 测试集样本数为61


# 逻辑回归模型
lr = LogisticRegression() 

# 随机森林模型
rf = RandomForestClassifier()

# 决策树模型
dt = DecisionTreeClassifier()

# 在训练集数据上进行模型的训练 
lr.fit(x_train, y_train)
rf.fit(x_train, y_train)
dt.fit(x_train, y_train)

# 在测试集得出准确率指标:
acc = lr.score(x_test, y_test )
print('-'*100)
print('逻辑回归: ', acc)
print('-'*100)

acc = rf.score(x_test, y_test )
print('-'*100)
print('随机森林: ', acc)
print('-'*100)

acc = dt.score(x_test, y_test )
print('-'*100)
print('决策树: ', acc)
print('-'*100)

# 交叉验证的结果更可信
from sklearn.model_selection import cross_val_score
print("*"*50)    
basari=cross_val_score(estimator=lr,X=x,y=y,cv=5)
print('逻辑回归: ', basari.mean())
print("*"*50) 


print("*"*50)    
basari=cross_val_score(estimator=rf,X=x,y=y,cv=5)
print('随机森林: ', basari.mean())
print("*"*50) 

print("*"*50)    
basari=cross_val_score(estimator=dt,X=x,y=y,cv=5)
print('决策树: ', basari.mean())
print("*"*50) 
