from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import pandas as pd
import kagglehub
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
torch.set_num_threads(1)

# Download latest version
path = kagglehub.dataset_download("jtrotman/formula-1-race-data")

results = pd.read_csv(f"{path}/results.csv", na_values="\\N")
races = pd.read_csv(f"{path}/races.csv", na_values="\\N")
drivers = pd.read_csv(f"{path}/drivers.csv", na_values="\\N")
teams = pd.read_csv(f"{path}/constructors.csv", na_values="\\N")

df = (results
      .merge(races[["raceId", "year", "round", "name", "date"]], on="raceId")
      .merge(drivers[["driverId", "surname", "code"]], on="driverId")
      .merge(teams[["constructorId", "name"]].rename(columns={"name": "team"}), on="constructorId")
      .sort_values(["year", "round"]))

df = df[df["year"] >= 2018]

y = df["positionOrder"]


X = pd.get_dummies(df[["grid", "team"]], columns=["team"], drop_first=True)

X_train, X_test, y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state= 42)

impute = SimpleImputer(strategy="most_frequent")
impute.fit(X_train)
X_train_imputed = impute.transform(X_train)
X_test_imputed = impute.transform(X_test)

scaler = StandardScaler()
scaler.fit(X_train_imputed)
X_train_imputed_scaled = scaler.transform(X_train_imputed)
X_test_imputed_scaled = scaler.transform(X_test_imputed)

kfold = KFold(n_splits = 10, shuffle = True, random_state = 42)

print("----LinearRegression----")
line = LinearRegression()
line.fit(X_train_imputed_scaled, y_train)
print ("score: ", line.score(X_test_imputed_scaled, y_test))
print("prediction: ", line.predict(X_test_imputed_scaled)[:10].round(0))

print()
print("----KNN----")
knn = KNN ()                 
grids = {"n_neighbors": [1,5,10,15,20]}   
gscv = GridSearchCV(knn, grids, cv = kfold, n_jobs = -1, scoring = "r2")               
gscv .fit(X_train_imputed_scaled, y_train)          
print(gscv.best_params_)         
print ("score: ", gscv .score(X_test_imputed_scaled, y_test))
print("prediction: ", gscv.predict(X_test_imputed_scaled)[:10].round(0))                                 
                                                           
print()                                                    
print("----DecisionTreeRegressor----")                                       
tree = DecisionTreeRegressor () 
grids = {"min_samples_leaf":[1,5,10,15,20]}
gscv2 = GridSearchCV(tree, grids, cv = kfold , n_jobs= -1, scoring = "r2")                            
gscv2.fit(X_train_imputed_scaled, y_train)
print(gscv2.best_params_)                      
print ("score: ", gscv2.score(X_test_imputed_scaled, y_test))
print("prediction: ", gscv2.predict(X_test_imputed_scaled)[:10].round(0))

print()
print("----RandomForestRegressor----")                                       
fr = RandomForestRegressor(n_estimators=50, min_samples_leaf = 15, criterion='squared_error' , random_state= 42)                                               
fr.fit(X_train_imputed_scaled, y_train)                               
print ("score: ", fr.score(X_test_imputed_scaled, y_test))            
print("prediction: ", fr.predict(X_test_imputed_scaled)[:10].round(0))
                                      
                                      
print()                               
print("----Neural Network----")

device = torch.device("cpu")

def t(a):
    return torch.tensor(a, dtype=torch.float32)

X_tr, X_val, y_tr, y_val = train_test_split(X_train_imputed_scaled, y_train.values, test_size=0.25, random_state=42)

X1_train, X1_val, X1_test = t(X_tr), t(X_val), t(X_test_imputed_scaled)
y1_train = t(y_tr).view(-1, 1)
y1_val = t(y_val).view(-1, 1)
y1_test = t(y_test.values).view(-1, 1)

torch.manual_seed(42)

model = nn.Sequential()
model.add_module("Linear1", nn.Linear(18, 256))
model.add_module("tanh1", nn.Tanh())
model.add_module("Linear2", nn.Linear(256, 256))
model.add_module("tanh2", nn.Tanh())
model.add_module("Linear3", nn.Linear(256, 1))

model.to(device)

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    y_hat = model(X1_train)
    loss_value = loss(y_hat, y1_train)
    loss_value.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        y_val_pred = model(X1_val)
        val_loss = loss(y_val_pred, y1_val)

    if epoch % 100 == 0:
        print(epoch, "train:", loss_value.item(), "val:", val_loss.item())

with torch.no_grad():
    y_pred = model(X1_test)
    pred_loss = loss(y_pred, y1_test)

print("NN score:", r2_score(y1_test.numpy(), y_pred.numpy()))
print("prediction", y_pred.numpy().ravel()[:10].round(0))