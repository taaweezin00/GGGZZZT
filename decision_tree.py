import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# โหลดชุดข้อมูล
data = pd.read_csv('myopia_prediction_data.csv')

# แปลงค่าของ Gender เป็นตัวเลข (0 = Male, 1 = Female)
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# แยก Features และ Target รวมทุกฟีเจอร์ที่เกี่ยวข้อง
X = data[['Age', 'Gender', 'ScreenTimeHoursPerDay', 'HasFamilyHistory', 'OutdoorTimeHoursPerDay']]
y = data['Myopia']

# แบ่งข้อมูลเป็นชุดฝึกและทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=16)

# สร้างโมเดล Decision Tree
DT_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=16)

# ฝึกโมเดล
DT_model.fit(X_train, y_train)

# ทำนายผล
y_pred = DT_model.predict(X_test)

# ประเมินความแม่นยำ
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# บันทึกโมเดลด้วย pickle
with open('myopia_model.pkl', 'wb') as f:
    pickle.dump(DT_model, f)

print("โมเดลถูกบันทึกลงไฟล์ 'myopia_model.pkl' เรียบร้อยแล้ว")
