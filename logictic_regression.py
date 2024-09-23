from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_full = pd.read_csv("full_dataset.csv")

mqn_column_rename = {
    'MQN1': 'Carbon atoms',
    'MQN2': 'Fluorine atoms',
    'MQN3': 'Chlorine atoms',
    'MQN4': 'Bromine atoms',
    'MQN5': 'Iodine atoms',
    'MQN6': 'Sulphur atoms',
    'MQN7': 'Phosphor atoms',
    'MQN8': 'Acyclic nitrogen atoms',
    'MQN9': 'Cyclic nitrogen atoms',
    'MQN10': 'Acyclic oxygen atoms',
    'MQN11': 'Cyclic oxygen atoms',
    'MQN12': 'Heavy atom count',

    'MQN13': 'Acyclic single bonds',
    'MQN14': 'Acyclic double bonds',
    'MQN15': 'Acyclic triple bonds',
    'MQN16': 'Cyclic single bonds',
    'MQN17': 'Cyclic double bonds',
    'MQN18': 'Cyclic triple bonds',
    'MQN19': 'Rotatable bonds',

    'MQN20': 'H-Bond donor atoms',
    'MQN21': 'H-Bond donor sites',
    'MQN22': 'H-Bond acceptor atoms',
    'MQN23': 'H-Bond acceptor sites',
    'MQN24': 'Positive charges',
    'MQN25': 'Negative charges',

    'MQN26': 'Acyclic monovalent nodes',
    'MQN27': 'Acyclic divalent nodes',
    'MQN28': 'Acyclic trivalent nodes',
    'MQN29': 'Acyclic tetravalent nodes',
    'MQN30': 'Cyclic divalent nodes',
    'MQN31': 'Cyclic trivalent nodes',
    'MQN32': 'Cyclic tetravalent nodes',
    'MQN33': '3-Membered rings',
    'MQN34': '4-Membered rings',
    'MQN35': '5-Membered rings',
    'MQN36': '6-Membered rings',
    'MQN37': '7-Membered rings',
    'MQN38': '8-Membered rings',
    'MQN39': '9-Membered rings',
    'MQN40': '≥ 10 membered rings',
    'MQN41': 'Atoms shared by fused rings',
    'MQN42': 'Bonds shared by fused rings'
}

df_full.rename(columns=mqn_column_rename, inplace=True)

df = df_full.drop(['SMR', 'cid', 'Molecule (RDKit Mol)'], axis=1)

X = df.drop('is_psychedelic', axis=1)
y = df['is_psychedelic']

# Разделение данных на обучающую, валидационную и тестовую выборки
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

# Масштабирование данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Логистическая регрессия с L1-регуляризацией для отбора признаков
lasso_log_reg = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
lasso_log_reg.fit(X_train_scaled, y_train)

# SelectFromModel для отбора признаков на основе L1-регуляризации
selector = SelectFromModel(lasso_log_reg, prefit=True)
selected_features_mask = selector.get_support()

# Применяем маску к исходным именам столбцов, чтобы узнать, какие признаки были отобраны
selected_features = X.columns[selected_features_mask]
print("Отобранные признаки:")
print(selected_features)

# Преобразуем данные, используя отобранные признаки
X_train_selected = selector.transform(X_train_scaled)
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)

# Логистическая регрессия с L2-регуляризацией на отобранных признаках
log_reg = LogisticRegression(penalty='l2', class_weight='balanced', random_state=42, C=0.001)

# Кросс-валидация на отобранных признаках (с метрикой ROC AUC)
cv_scores = cross_val_score(log_reg, X_train_selected, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
print(f"Средний ROC AUC на кросс-валидации: {np.mean(cv_scores)}")

# Обучаем модель на всех тренировочных данных после кросс-валидации
log_reg.fit(X_train_selected, y_train)

# Оценка модели на валидационной выборке
y_val_pred = log_reg.predict(X_val_selected)
y_val_prob = log_reg.predict_proba(X_val_selected)[:, 1]
roc_auc_val = roc_auc_score(y_val, y_val_prob)
print(f"ROC AUC на валидационной выборке: {roc_auc_val}")

# Оценка модели на тестовой выборке
y_test_pred = log_reg.predict(X_test_selected)
y_test_prob = log_reg.predict_proba(X_test_selected)[:, 1]
roc_auc_test = roc_auc_score(y_test, y_test_prob)

print("Classification report on test data:")
print(classification_report(y_test, y_test_pred))
print(f"ROC AUC на тестовой выборке: {roc_auc_test}")

# Матрица ошибок
print("Матрица ошибок на тестовой выборке:")
print(confusion_matrix(y_test, y_test_pred))

def plot_learning_curve(model, X, y, title="Learning Curve"):
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10)
    )

    # Усреднение значений по folds
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    # Заполнение области вокруг кривых для отображения вариации
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, valid_mean, 'o-', color="g", label="Validation score")

    plt.legend(loc="best")
    plt.show()

plot_learning_curve(log_reg, X_train_selected, y_train, title="Learning Curve (Logistic Regression)")