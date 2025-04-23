from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def train_model(X_train, y_train, model_name="rf"):
    if model_name == "logreg":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "svm":
        model = SVC(kernel='rbf', probability=True)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model