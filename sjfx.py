# -*- coding:utf-8 -*-
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 构造CarSeat数据集（离线版本，不需要联网下载）
data = {
    "Sales": [7.5,12.5,10.5,9.5,11.2,12.8,11.5,13.1,8.7,10.9]*10,
    "Price": [120,110,125,130,115,105,118,102,135,122]*10,
    "Income": [45,62,58,70,52,68,55,75,42,60]*10,
    "Advertising": [12,15,10,8,14,18,11,20,7,13]*10,
    "ShelveLoc": ["Bad","Good","Medium","Bad","Good","Medium","Bad","Good","Medium","Bad"]*10
}
df = pd.DataFrame(data)

print("数据集前5行：")
print(df.head())

# 选取变量：响应变量Sales；自变量Price, Income, Advertising, ShelveLoc
X_raw = df[["Price", "Income", "Advertising", "ShelveLoc"]]
y = df["Sales"]

# 生成虚拟变量，ShelveLoc是分类变量
X_dummy = pd.get_dummies(X_raw, columns=["ShelveLoc"], drop_first=True)

# 添加常数项(截距β0)，statsmodels OLS需要手动加
X = sm.add_constant(X_dummy)

# 构建多元线性回归模型，拟合
model = sm.OLS(y, X).fit()

# 输出回归结果
print("\n=====回归结果摘要=====")
print(model.summary())

# ========== 计算VIF（方差膨胀因子，检验多重共线性）==========
vif_data = pd.DataFrame()
vif_data["变量名"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\n=====方差膨胀因子VIF=====")
print(vif_data)
