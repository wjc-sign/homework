# ===================== 导入所需库 =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error

# 设置Matplotlib中文显示，解决负号乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 读取并预处理Hitters棒球数据集 =====================
# 读取csv文件，r原始字符串防止路径转义错误
df = pd.read_csv(r"D:\Chat records\QQ Downloads\python数据分析\hitters.csv")

# 查看数据集基础信息：样本量、列名、数据类型、缺失值
print("数据集行列数：", df.shape)
print("\n数据集信息：")
print(df.info())

# 目标变量Salary存在缺失，删除目标变量为空的样本
df = df.dropna(subset=["Salary"])

# 将分类变量League、Division、NewLeague自动转为哑变量，drop_first删除基准组，规避多重共线性
df = pd.get_dummies(df, drop_first=True)

# 划分特征矩阵X与目标变量y（薪资Salary）
y = df["Salary"]
X = df.drop("Salary", axis=1)

# 划分训练集70%，测试集30%；random_state固定随机种子保证结果可复现
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 正则化模型（Ridge/Lasso/ElasticNet）必须标准化，消除特征量纲影响
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit仅用训练集学习均值方差，防止数据泄露
X_test_scaled = scaler.transform(X_test)        # 测试集直接使用训练集的均值方差

# ===================== 2. 建立模型，10折交叉验证自动寻找最优alpha =====================
# Ridge：L2正则，仅压缩系数，不会把系数压缩到0
# alphas：生成对数等间隔的正则强度候选值
ridge = RidgeCV(alphas=np.logspace(-3, 5, 100), cv=10)
ridge.fit(X_train_scaled, y_train)

# Lasso：L1正则，可以直接将不重要特征系数压缩至0，实现特征筛选
lasso = LassoCV(alphas=np.logspace(-3,5,100), cv=10, max_iter=20000)
lasso.fit(X_train_scaled, y_train)

# ElasticNet：弹性网，L1+L2混合正则，兼顾Ridge和Lasso优点
# l1_ratio：L1惩罚占比，0=纯Ridge，1=纯Lasso
enet = ElasticNetCV(
    alphas=np.logspace(-3,5,100),
    l1_ratio=[0.1,0.3,0.5,0.7,0.9],
    cv=10, max_iter=20000
)
enet.fit(X_train_scaled, y_train)

# ===================== 3. 模型评估：定义函数计算RMSE =====================
def get_rmse(model, Xtest, ytest):
    """
    输入训练好的模型、测试特征、测试标签
    返回测试集RMSE（均方根误差，误差越小预测效果越好）
    """
    y_pred = model.predict(Xtest)
    return np.sqrt(mean_squared_error(ytest, y_pred))

# 计算三个模型在测试集上的RMSE
ridge_rmse = get_rmse(ridge, X_test_scaled, y_test)
lasso_rmse = get_rmse(lasso, X_test_scaled, y_test)
enet_rmse = get_rmse(enet, X_test_scaled, y_test)

# 打印最优超参数与测试集RMSE结果
print("\n" + "="*70)
print(f"Ridge 最优alpha = {ridge.alpha_:.4f}, 测试集RMSE = {ridge_rmse:.2f}")
print(f"Lasso 最优alpha = {lasso.alpha_:.4f}, 测试集RMSE = {lasso_rmse:.2f}")
print(f"ElasticNet 最优alpha = {enet.alpha_:.4f}, l1_ratio={enet.l1_ratio_:.2f}, 测试集RMSE={enet_rmse:.2f}")
print("="*70)

# 整合三个模型的所有回归系数，方便对比查看
coef_result = pd.DataFrame({
    "Feature": X.columns,
    "Ridge_coef": ridge.coef_,
    "Lasso_coef": lasso.coef_,
    "ElasticNet_coef": enet.coef_
})
print("\n各模型回归系数对比：")
print(coef_result)

# 统计非零系数数量（阈值1e-5，排除极小浮点误差）
print("\n==== 各模型非零特征数量 ====")
print(f"Ridge非零系数：{np.sum(np.abs(ridge.coef_)>1e-5)}")
print(f"Lasso非零系数：{np.sum(np.abs(lasso.coef_)>1e-5)}")
print(f"ElasticNet非零系数：{np.sum(np.abs(enet.coef_)>1e-5)}")

# ===================== 4. 绘制Ridge、Lasso系数路径图（Coefficient Path） =====================
# alpha候选值，和前面CV搜索范围保持一致
alphas = np.logspace(-3, 5, 100)

# 遍历不同alpha，保存Ridge模型每一组系数
ridge_coefs = []
for a in alphas:
    m = Ridge(alpha=a)
    m.fit(X_train_scaled,y_train)
    ridge_coefs.append(m.coef_)

# 遍历不同alpha，保存Lasso模型每一组系数
lasso_coefs = []
for a in alphas:
    m = Lasso(alpha=a, max_iter=20000)
    m.fit(X_train_scaled,y_train)
    lasso_coefs.append(m.coef_)

# 转为numpy数组方便绘图
ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

# 绘制子图：左Ridge，右Lasso
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
plt.plot(alphas, ridge_coefs)
plt.xscale("log")   # x轴对数坐标，方便观察大范围alpha
plt.title("Ridge 系数路径图")
plt.xlabel(r"正则化强度 $\alpha$")
plt.ylabel("回归系数值")

plt.subplot(1,2,2)
plt.plot(alphas, lasso_coefs)
plt.xscale("log")
plt.title("Lasso 系数路径图")
plt.xlabel(r"正则化强度 $\alpha$")
plt.ylabel("回归系数值")

plt.tight_layout() # 自动调整子图间距，防止文字重叠
plt.show()
