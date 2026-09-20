import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
df = pd.read_csv("D:\ records\QQ Downloads\python\Carseats.csv")


formula = "Sales ~ Price + Income + Advertising + ShelveLoc"
model = smf.ols(formula=formula, data=df).fit()


print(model.summary())

print("\nShelveLoc的基准组：Bad")
# statsmodels formula会自动把字母排序第一个类别Bad作为参照基准组，生成ShelveLoc[T.Good], ShelveLoc[T.Medium]

print("\nShelveLoc[Good]系数含义：")
print(f"ShelveLoc[T.Good]系数 = {model.params['ShelveLoc[T.Good]']:.4f}")
print("在Price、Income、Advertising保持不变的前提下，货架位置为Good的商品，对比基准组Bad，销售额Sales平均增加该系数大小。")

# ==========3. 计算VIF，多重共线性诊断==========
# 构造模型设计矩阵（去掉截距项计算VIF）
X = model.model.exog
vif_df = pd.DataFrame()
vif_df["变量名"] = model.model.exog_names
vif_df["VIF"] = [variance_inflation_factor(X,i) for i in range(X.shape[1])]
print("\n各变量VIF结果：")
print(vif_df)
