driverpath="./data"

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import re

print("read data")
v = pd.read_csv(f"{driverpath}/product_u03.csv")

url = 'https://amis.afa.gov.tw/veg/VegProdDayTransInfo.aspx'

ses = requests.Session()

r = ses.post(url)

# 此函式會找特定的value，如「__VIEWSTATE」等
def find_value(name, web):
    reg = 'name="' + name + '".+value="(.*)" />'
    pattern = re.compile(reg)
    result  = pattern.findall(web.text)
    try:
        return result[0]
    except:
        return ""

# 使用方式
__VIEWSTATE = find_value('__VIEWSTATE', r)

__VIEWSTATEGENERATOR = find_value('__VIEWSTATEGENERATOR', r)

__EVENTVALIDATION = find_value('__EVENTVALIDATION', r)

r_txt = BeautifulSoup(r.text, "html.parser")

headers = {
      'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
      'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/32.0.1664.3 Safari/537.36'
    }

def normalize(train):
    train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    return train_norm

def LSTM_model():
  model = keras.models.Sequential([
              keras.layers.Input(shape=(5,1)),
              keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
              keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
              keras.layers.Bidirectional(keras.layers.LSTM(25, return_sequences=False)),
              keras.layers.Dense(1),])
              #keras.layers.Dense(128, activation='selu'),
  return model

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping

data = {
    'ctl00$ScriptManager_Master': 'ctl00$ScriptManager_Master|ctl00$contentPlaceHolder$btnQuery',
    'ctl00$contentPlaceHolder$ucDateScope$rblDateScope': 'P',
    'ctl00$contentPlaceHolder$ucSolarLunar$radlSolarLunar': 'S',
    'ctl00$contentPlaceHolder$txtSTransDate': '103/03/27',
    'ctl00$contentPlaceHolder$txtETransDate': '113/03/30',
    'ctl00$contentPlaceHolder$txtMarket': '台北一',
    'ctl00$contentPlaceHolder$hfldMarketNo': '109',
    'ctl00$contentPlaceHolder$txtProduct': 'LA1 甘藍 初秋',
    'ctl00$contentPlaceHolder$hfldProductNo': 'LA1',
    'ctl00$contentPlaceHolder$hfldProductType': 'S',
    '__EVENTTARGET': '',
    '__EVENTARGUMENT': '',
    '__VIEWSTATE': __VIEWSTATE,
    '__VIEWSTATEGENERATOR': __VIEWSTATEGENERATOR,
    '__EVENTVALIDATION': __EVENTVALIDATION,
    '__ASYNCPOST': 'true',
    'ctl00$contentPlaceHolder$btnQuery': '查詢'
}

page = ses.post(url,headers=headers,data=data)
r_txt = BeautifulSoup(page.text, "html.parser")
tables = r_txt.findAll('table')
tr = tables[2].findAll('tr')

tds = []
for raw in tr:
     data = [td.get_text() for td in raw.findAll("td")]
     tds.append(data)

data_temp = pd.DataFrame(tds[2:],columns=tds[0])
data_temp2 = data_temp.dropna().reset_index(drop=True)
df_temp = data_temp2.drop(columns=['日期', '產品','跟前一交易日比較%','跟前一交易日比較%'])
df_temp['交易量(公斤)'] = df_temp['交易量(公斤)'].str.replace(',','')
df_temp.rename(columns={'上價': 'highest price', '中價': 'median price','下價':'lowest price','平均價(元/公斤)':'average price','交易量(公斤)':'volume'}, inplace=True)

df_temp = df_temp.apply(pd.to_numeric)

def normalize(train):
    train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    return train_norm

df_temp_norm = normalize(df_temp)

avg_all_price = []
df_price = pd.DataFrame()
print('run_train_f')

def train_f(state):
  if state == "train":
    for i in range(0,len(v)):
      v_id = v.iloc[i,0]
      v_detail = v.iloc[i,1]
      data = {
        'ctl00$ScriptManager_Master': 'ctl00$ScriptManager_Master|ctl00$contentPlaceHolder$btnQuery',
        'ctl00$contentPlaceHolder$ucDateScope$rblDateScope': 'P',
        'ctl00$contentPlaceHolder$ucSolarLunar$radlSolarLunar': 'S',
        'ctl00$contentPlaceHolder$txtSTransDate': '103/03/27',
        'ctl00$contentPlaceHolder$txtETransDate': '113/03/30',
        'ctl00$contentPlaceHolder$txtMarket': '台北一',
        'ctl00$contentPlaceHolder$hfldMarketNo': '109',
        'ctl00$contentPlaceHolder$txtProduct': v_detail,
        'ctl00$contentPlaceHolder$hfldProductNo': v_id,
        'ctl00$contentPlaceHolder$hfldProductType': 'S',
        '__EVENTTARGET': '',
        '__EVENTARGUMENT': '',
        '__VIEWSTATE': __VIEWSTATE,
        '__VIEWSTATEGENERATOR': __VIEWSTATEGENERATOR,
        '__EVENTVALIDATION': __EVENTVALIDATION,
        '__ASYNCPOST': 'true',
        'ctl00$contentPlaceHolder$btnQuery': '查詢'
        }
      page = ses.post(url,headers=headers,data=data)
      r_txt = BeautifulSoup(page.text, "html.parser")
      tables = r_txt.findAll('table')
      tr = tables[2].findAll('tr')
      tds = []
      for raw in tr:
        data = [td.get_text() for td in raw.findAll("td")]
        tds.append(data)
      data = pd.DataFrame(tds[2:],columns=tds[0])
      data2 = data.dropna().reset_index(drop=True)
      df = data2.drop(columns=['日期', '產品','跟前一交易日比較%','跟前一交易日比較%'])
      df['交易量(公斤)'] = df['交易量(公斤)'].str.replace(',','')
      df.rename(columns={'上價': 'highest price', '中價': 'median price','下價':'lowest price','平均價(元/公斤)':'average price','交易量(公斤)':'volume'}, inplace=True)
      df_factor = df

      split_day = 720
      actual_price = df_factor['average price'].iloc[len(df_factor)-split_day:]
      multi_data = df_factor.iloc[len(df_factor)-split_day:]
      test_data = df_factor.iloc[len(df_factor)-split_day:]
      train_data = df_factor.iloc[:len(df_factor)-split_day]
      test_data.reset_index(drop = True,inplace = True)
      train_data.reset_index(drop = True,inplace = True)

      temp_data = []

      temp_actual_price = []
      for i in range(len(train_data)):
        if train_data['average price'].values[i] != 0:
          temp_data.append(train_data.iloc[i])
      train_data = temp_data
      temp_data2 = []
      for i in range(len(test_data)):
        if test_data['average price'].values[i] != 0:
          temp_data2.append(test_data.iloc[i])

      for i in range(len(actual_price)):
        if actual_price.values[i] != 0:
          temp_actual_price.append(actual_price.values[i])

      test_data = temp_data2
      actual_price = temp_actual_price

      data = pd.DataFrame(train_data)
      data2 = pd.DataFrame(test_data)

      day = 240
      #data_tmp = data.copy()
      data_tmp = data[0:-day]

      #data_tmp2 = data2.copy()
      data_tmp2 = data2[0:-day]

      if len(data) == 0:
        avg_all_price.append(0)
        continue

      temp = data['average price'].iloc[day:]
      temp2 = data2['average price'].iloc[day:]
      temp = temp.reset_index(drop=True)
      data_tmp = data_tmp.reset_index(drop=True)
      temp2 = temp2.reset_index(drop=True)
      data_tmp2 = data_tmp2.reset_index(drop=True)

      temp.name = 'price after ' + str(day) + ' days'
      temp2.name = 'price after ' + str(day) + ' days'
      data = pd.concat([data_tmp,temp], axis=1)
      data2 = pd.concat([data_tmp2,temp2], axis=1)
      prediction_days = day
      data = data.apply(pd.to_numeric)
      data2 = data2.apply(pd.to_numeric)
      dataset_norm = normalize(data)
      dataset_norm2 = normalize(data2)
      dataset_norm = pd.concat([dataset_norm, dataset_norm2])

      X_train = dataset_norm.iloc[:,0:5]
      y_train = dataset_norm.iloc[:,5]

      model = LSTM_model()
      model.compile(optimizer='adam', loss='mean_squared_error')
      model.fit(X_train,y_train,epochs=50,batch_size = 100)

      #model.save(f'/content/drive/MyDrive/Vag2024/{v.iloc[i,0]}_model.h5')
      X_test = dataset_norm2.iloc[:,0:5]
      y_test = dataset_norm2.iloc[:,5]

      data_min = np.min(data2['average price'])
      data_max = np.max(data2['average price'])
      data_mean = np.mean(data2['average price'])
      predictions = model.predict(X_test)

      Y_org = y_test*(data_max-data_min) + data_mean
      predictions_org = predictions*(data_max-data_min) + data_mean
      price_pre = predictions_org[-day:]
      price_pre2 = price_pre.reshape((day,))
      df_price[v_detail] = price_pre2.tolist()

      price_avg = np.average(price_pre)
      print(price_avg)
      avg_all_price.append(price_avg)
    df_price.to_csv(f"{driverpath}/predict_price.csv",index=False)
  else:
    return

train_f("val")

df_price = pd.read_csv(f"{driverpath}/predict_price.csv")

df_price.columns = v["原生"].values.tolist()

df_price_avg = df_price.groupby(pd.cut(df_price.index, [0, 20, 40, 60, 80,100,120,140,160,180,200,220,240])).mean()
df_price_avg.reset_index(drop=True,inplace=True)
df_price_avg = df_price_avg.round(2)
df_price_avg.insert(loc=0, column="月份", value=["下一個月","下二個月","下三個月","下四個月","下五個月","下六個月","下七個月","下八個月","下九個月","下十個月","下十一個月","下十二個月"])
df_price_avg

df_price_avg.to_csv(f"{driverpath}/predict_price_avg.csv",index=False)

df_price_avg = pd.read_csv(f"{driverpath}/predict_price_avg.csv")

month = "下一個月"

avg_all_price = df_price_avg[df_price_avg["月份"]==month].iloc[:,1:].values[0]

avg_all_price

v["value"] = avg_all_price



v.to_csv(f"{driverpath}/r.csv",index=False)

v = pd.read_csv(f"{driverpath}/r.csv")



def nodejson():

  str_s1 = '{"nodes": ['
  s1 = ''
  s2 = ''
  for i in range(0,len(v)):

    if i == (len(v)-1):
      s1 = s1 + '{"id": "'+v.iloc[i,2]+' $'+str(v.iloc[i,4])+'", "group": '+str(v.iloc[i,4])+'}'
    else:
      s1 = s1 + '{"id": "'+v.iloc[i,2]+' $'+str(v.iloc[i,4])+'", "group": '+str(v.iloc[i,4])+'},'

  str_s2 = '],"links": ['

  for i in range(0,len(v)):
    if pd.isna(v.iloc[i,3]):
      pass
    else:
      sp = v.iloc[i,3].split(",")
    for j in range(0,len(sp)):
      if (i == (len(v)-1)) and (j == (len(sp)-1)):
        s2 = s2 + '{"source": "'+v.iloc[i,2]+' $'+str(v.iloc[i,4])+'", "target": "'+sp[j]+' $'+str(v[v["原生"]==sp[j]]["value"].iloc[0])+'", "value":'+ str(v[v["原生"]==sp[j]]["value"].iloc[0])+'}'
      else:
        s2 = s2 + '{"source": "'+v.iloc[i,2]+' $'+str(v.iloc[i,4])+'", "target": "'+sp[j]+' $'+str(v[v["原生"]==sp[j]]["value"].iloc[0])+'", "value":'+ str(v[v["原生"]==sp[j]]["value"].iloc[0])+'},'

  str_s3 = ']}'

  return str_s1 + s1+str_s2 + s2 + str_s3

sel = input("原生：")

v[v["原生"]==sel]["value"].iloc[0]

sub_price = []
sub_name = []
for i in range(0,len(v)):
  if pd.isna(v.iloc[i,3]):
    pass
  elif v.iloc[i,2] == sel:
    sp = v.iloc[i,3].split(",")
    for j in range(0,len(sp)):
      sub_price.append(v[v["原生"]==sp[j]]["value"].iloc[0])
      sub_name.append(sp[j])

sub_price

sub_name

df_sub = pd.DataFrame()
df_sub["替代"] = sub_name
df_sub["預測"] = sub_price



df_sub_sort = df_sub.sort_values(by=['預測'], ascending=False)


df_sub_sort_filter = df_sub_sort.loc[df_sub_sort['預測'] > v[v["原生"]==sel]["value"].iloc[0]]


# !pip install pyngrok
from flask import Flask, request, render_template
from pyngrok import ngrok

# !ngrok authtoken 2dZk3GAUUq1OnA3aHZGNtt1vGuo_3VjSy4B8CyWrU9rvoiVta

app = Flask(__name__,template_folder=f"./templates")
port = 5000
public_url = ngrok.connect(port).public_url
app.config["BASE_URL"] = public_url
print(public_url)



ki = v["原生"].values.tolist()


mon = df_price_avg["月份"].values.tolist()

nodejson()

@app.route('/data',methods=["GET"])
def rawdata():
  return render_template("table.html", column_names=df_temp.columns.values, row_data=list(df_temp.values.tolist()),link_column="Patient ID", zip=zip)

@app.route('/datanor',methods=["GET"])
def nordata():
  return render_template("table.html", column_names=df_temp_norm.columns.values, row_data=list(df_temp_norm.values.tolist()),link_column="Patient ID", zip=zip)

@app.route('/datapre',methods=["GET"])
def predata():
  return render_template("table.html", column_names=df_price_avg.columns.values, row_data=list(df_price_avg.values.tolist()),link_column="Patient ID", zip=zip)

@app.route('/datasel',methods=["GET"])
def seldata():
  ki = v["原生"].values.tolist()
  mon = df_price_avg["月份"].values.tolist()
  return render_template("selection.html", months=mon, kinds=ki,link_column="Patient ID", zip=zip)

@app.route('/select',methods=["POST"])
def resultdata():
  month = request.form.get('month')
  sel = request.form.get('kind')
  avg_all_price = df_price_avg[df_price_avg["月份"]==month].iloc[:,1:].values[0]
  v["value"] = avg_all_price
  datajson = nodejson()
  sub_price = []
  sub_name = []
  for i in range(0,len(v)):
    if pd.isna(v.iloc[i,3]):
      pass
    elif v.iloc[i,2] == sel:
      sp = v.iloc[i,3].split(",")
      for j in range(0,len(sp)):
        sub_price.append(v[v["原生"]==sp[j]]["value"].iloc[0])
        sub_name.append(sp[j])
  df_sub = pd.DataFrame()
  df_sub["替代"] = sub_name
  df_sub["預測"] = sub_price
  df_sub_sort = df_sub.sort_values(by=['預測'], ascending=False)
  df_sub_sort_filter = df_sub_sort.loc[df_sub_sort['預測'] > v[v["原生"]==sel]["value"].iloc[0]]

  return render_template("table2.html", column_names1=df_sub_sort.columns.values, row_data1=list(df_sub_sort.values.tolist()),column_names2=df_sub_sort_filter.columns.values, row_data2=list(df_sub_sort_filter.values.tolist()), nodedata=datajson ,link_column="Patient ID", zip=zip)

app.run()

