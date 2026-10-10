# ======================================================
# 项目：Netflix用户行为数据 ElasticNet弹性网回归
# 任务：基于用户行为特征，预测用户平均打分 avg_rating_given
# 模型：ElasticNet（L1+L2混合正则，自动特征筛选+防过拟合）
# 技术：Pipeline流水线，一次性完成【预处理+建模】
# ======================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 设置中文显示，解决负号乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入sklearn工具包
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error

# ---------------------- 1. 读取并清洗数据 ----------------------
# 读取Netflix用户行为数据集
df = pd.read_csv(r"D:\Chat records\QQ Downloads\python数据分析\netflix_user_behavior_churn_50000.csv")

# 定义预测目标：用户平均打分
target_col = "avg_rating_given"

# 删除目标变量存在缺失的样本（目标缺失样本无法用于训练）
df = df.dropna(subset=[target_col])

# 划分标签y与原始特征X
# 丢弃目标列与 churned流失标签，本次不做流失预测
y = df[target_col].copy()
X_raw = df.drop([target_col, "churned"], axis=1)

# 区分数值型特征、类别型特征，分别做预处理
num_cols = [
    "account_age_months",                # 账号使用时长(月)
    "session_count",                     # 会话总数
    "avg_watch_time_minutes_per_week",   # 每周平均观看时长(分钟)
    "watch_sessions_per_week",           # 每周观看次数
    "completion_rate",                   # 影片完成率
    "recommendation_click_rate",         # 推荐内容点击率
    "days_since_last_login",             # 距离上次登录天数
    "app_rating"                         # APP基础评分
]
cat_cols = [
    "favorite_genre",       # 偏好影片类型
    "time_of_day",          # 观看时段
    "recommendation_source" # 推荐来源
]

# ---------------------- 2. 构建预处理流水线 ----------------------
# ColumnTransformer：对不同列使用不同预处理方式
# 数值特征：Z-score标准化（均值0，方差1）
# 类别特征：OneHot独热编码，drop="first" 避免多重共线性
preprocessor = ColumnTransformer(
    transformers=[
        ("num_scaler", StandardScaler(), num_cols),
        ("cat_ohe", OneHotEncoder(sparse_output=False, drop="first"), cat_cols)
    ]
)

# ---------------------- 3. 划分训练集、测试集 ----------------------
# 7:3划分，random_state固定随机种子保证结果可复现
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.3, random_state=42
)

# ---------------------- 4. ElasticNet模型 + 10折交叉验证 ----------------------
# ElasticNetCV：内置交叉验证，自动搜索最优超参数
# l1_ratio=0 → Ridge(L2正则)；l1_ratio=1 → Lasso(L1正则)；0~1为混合正则
enet_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],  # L1权重比例候选
    alphas=np.logspace(-4, 2, 100),                 # 正则强度λ候选（对数间隔）
    cv=10,                                          # 10折交叉验证
    max_iter=20000,                                 # 最大迭代次数，保证收敛
    random_state=42
)

# 完整Pipeline：先做数据预处理，再送入弹性网模型
# 优点：避免数据泄露，预测新数据时直接传入原始数据即可
pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("elasticnet", enet_cv)
])

# 在训练集上拟合流水线（自动完成预处理+模型训练+超参搜索）
pipe.fit(X_train_raw, y_train)

# ---------------------- 5. 模型评估与结果输出 ----------------------
# 获取交叉验证选出的最优超参数
best_alpha = pipe.named_steps["elasticnet"].alpha_
best_l1ratio = pipe.named_steps["elasticnet"].l1_ratio_

print("="*60)
print(f"✅ 最优正则系数 alpha(λ): {best_alpha:.6f}")
print(f"✅ 最优L1权重 l1_ratio: {best_l1ratio:.3f}")

# 测试集预测，计算RMSE（均方根误差，衡量预测误差大小）
y_pred = pipe.predict(X_test_raw)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"✅ 测试集RMSE: {rmse:.4f}")

# 提取全部特征名称（数值特征 + OneHot展开后的类别特征）
ohe_transformer = pipe.named_steps["preprocess"].named_transformers_["cat_ohe"]
cat_feature_names = ohe_transformer.get_feature_names_out(cat_cols).tolist()
all_feature_names = num_cols + cat_feature_names

# 提取模型回归系数，构建Series方便查看
coefs = pipe.named_steps["elasticnet"].coef_
coef_series = pd.Series(coefs, index=all_feature_names)

# ElasticNet自带稀疏性：系数绝对值极小视为0，筛选有效特征
non_zero_coef = coef_series[np.abs(coef_series) > 1e-5].sort_values(key=abs, ascending=False)

print("\n📌 ElasticNet筛选得到的非零特征（按系数绝对值降序）：")
print(non_zero_coef)

# ---------------------- 6. 可视化：非零回归系数水平条形图 ----------------------
plt.figure(figsize=(10, 7))
non_zero_coef.plot(kind="barh")
plt.title("ElasticNet 非零回归系数 | Netflix 用户评分预测")
plt.xlabel("回归系数大小（正值促进打分，负值抑制打分）")
plt.tight_layout()  # 自动调整布局，防止标签被截断
plt.show()
