import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
x= pd.read_csv('/Users/ziadhossam/Library/Containers/net.whatsapp.WhatsApp/Data/tmp/documents/DD26D5AD-CEFF-4B36-A649-0083C8A474CB/Cleaned_Train_Data.csv')


target = "Heart Disease"
corr = x.corr(numeric_only=True)[target].sort_values(ascending=False)

plt.figure()
corr.drop(target).plot(kind='bar')
plt.title("Correlation with Heart Disease")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


top_features = corr.drop(target).abs().sort_values(ascending=False).head(5).index

print("Top 5 Features:")
print(top_features)


sns.set_theme(style="whitegrid")
g = sns.catplot(
    data=x,
    x='Chest pain type',
    hue='Heart Disease',
    col='Exercise angina',
    kind='count',
    palette='Set2'
)

g.set_axis_labels("Chest Pain Type", "Count")
g.set_titles("Exercise Angina: {col_name}")
plt.subplots_adjust(top=0.85)
g.fig.suptitle('Chest Pain Type vs Heart Disease')

plt.show()





X = x[['Age', 'BP']]
y = x['Heart Disease']


model = LogisticRegression()
model.fit(X, y)


plt.figure()
plt.scatter(X['Age'], X['BP'], c=y)


x_min, x_max = X['Age'].min() - 1, X['Age'].max() + 1
y_min, y_max = X['BP'].min() - 1, X['BP'].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)


Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.contour(xx, yy, Z)

plt.xlabel("Age")
plt.ylabel("BP")
plt.title("Decision Boundary (Age vs BP)")
plt.show()

