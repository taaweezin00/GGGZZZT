from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# โหลดโมเดลที่บันทึกไว้
with open('myopia_model.pkl', 'rb') as f:
    model = pickle.load(f)

# โหลดข้อมูลทั้งหมด
data = pd.read_csv('myopia_prediction_data.csv')
# แปลง DataFrame เป็น HTML
data_html = data.to_html(classes='dataframe', header="true", index=False)

@app.route('/')
def home():
    return render_template('index.html', tables=[data_html], titles=data.columns.values)

@app.route('/predict', methods=['POST'])
def predict():
    # รับข้อมูลจากฟอร์ม
    try:
        Age = float(request.form['Age'])
        Gender = request.form['Gender']
        ScreenTimeHoursPerDay = float(request.form['ScreenTimeHoursPerDay'])
        HasFamilyHistory = int(request.form['HasFamilyHistory'])
        OutdoorTimeHoursPerDay = float(request.form['OutdoorTimeHoursPerDay'])
    except ValueError:
        return render_template('index.html', prediction_text='กรุณากรอกข้อมูลให้ครบถ้วนและถูกต้อง', tables=[data_html], titles=data.columns.values)

    # แปลงค่าของ Gender เป็นตัวเลข (0 = Female, 1 = Male)
    if Gender.lower() == 'male':
        Gender = 1
    else:
        Gender = 0

    # เตรียมข้อมูลสำหรับทำนาย
    input_features = np.array([[Age, Gender, ScreenTimeHoursPerDay, HasFamilyHistory, OutdoorTimeHoursPerDay]])

    # ทำนายผล
    prediction = model.predict(input_features)
    if prediction[0] == 1:
        Myopia = 'มีภาวะสายตาสั้น'
        prediction_class = 'positive'
    else:
        Myopia = 'ไม่มีภาวะสายตาสั้น'
        prediction_class = 'negative'

    return render_template('index.html',
                           prediction_text='ผลลัพธ์: {}'.format(Myopia),
                           prediction_class=prediction_class,
                           tables=[data_html],
                           titles=data.columns.values)

if __name__ == '__main__':
    app.run(debug=True)
