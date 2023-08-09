pip install matplotlib
pip install pandas
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

# 展示文本；文本直接使用Markdown语法
st.markdown("# LTV计算器demo")
st.markdown("""
            #### V1版本
            """)


# 加入交互控件，如输入框
sec_ret = st.number_input("1日留存率为",0.2)
third_ret = st.number_input("2日留存率为",0.15)
four_ret = st.number_input("3日留存率为",0.13)
five_ret = st.number_input("4日留存率为",0.12)
need_days = st.number_input("预估天数为",15)
# six_ret = st.number_input("6日留存为", 0.235)
# seven_ret = st.number_input("7日留存为", 0.228)

arpu = st.number_input("APRU为",0.0000001, format="%0.7f")
cac= st.number_input("CAC为",0.00005, format="%0.5f")

rat=[sec_ret,third_ret,four_ret,five_ret]

# scipy的curve_fit函数进行曲线拟合,模型为指数衰减模型
# pkg和day为1-3的数据作为已知xdata和ydata,进行曲线拟合,然后基于得到的模型参数预测1-30天的留存率

def fit_retention(sub_df):
    # print(sub_df)
    
    # 采用指数衰减模型 
    def func(x, a, b, c):
        return a*np.exp(-b*x) + c

    # 提取1-3天的数据
    xdata =['1','2','3','4']
    ydata = rat

    # 曲线拟合
    popt, pcov = curve_fit(func, xdata, ydata,maxfev=100000000)  
    a, b, c = popt
    
    print('系数a:', a)
    print('系数b:', b)
    print('系数pcov:', pcov)

    # 根据模型预测1-30天留存率
    x = np.arange(1, need_days)
    predict_y = func(x, a, b, c) 
    
    return predict_y

rat_predict=fit_retention(pd.DataFrame(rat))

format_str = "{:.6f}"
st.write("LT预估为：",  format_str.format(np.sum(rat_predict)))
st.write("LTV预估为：",  format_str.format(np.sum(rat_predict)*arpu))
st.write("ROI预估为：",  format_str.format(np.sum(rat_predict)*arpu/cac))

# import pyplot form matplotlib as plt

x = range(1,need_days)
y1 = list(rat_predict)

def cumulative_sum(values):
    total = 0
    cum_values = []
    
    for value in values:
        total += value
        cum_values.append(total)
        
    return cum_values

y2 = cumulative_sum(list(rat_predict*arpu)) #需要计算累计LTV



# 显示可交互数据帧
columns = ['retention_days','rate','cumulative_ltv']
dates = x # 留存日期数据
retention_rates = y1 # 留存率数据 
cumulative_ltv = y2 # 累计LTV数据

# # 定义格式化函数 
# def format_decimal(col, digits=2):
#     return '{{:{}.{}f}}'.format(col, digits)


table = pd.DataFrame({
    'retention_days': dates,
    'retention_rate_predict': retention_rates, 
    'cumulative_ltv': cumulative_ltv
}).set_index('retention_days')

# 应用格式化  
table['retention_rate_predict'] = table['retention_rate_predict'].round(2)
table['cumulative_ltv'] = table['cumulative_ltv'].round(6)

# print(table)
st.dataframe(table,900,527)  # Same as st.write(df)

# 绘制留存率和LTV图例
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()


# plt.figure(figsize=(10,5))
ax1.plot(x, y1, color='#1172D2', marker='^', linewidth=2, label='retention_rate')
ax2.plot(x, y2, color='#0A3447', marker='x', linewidth=2, label='cumulative_ltv')

plt.title('retention_rate and ltv')
plt.xlabel('retention_days')  
plt.ylabel('retention_rate and ltv')
ax1.legend()
ax2.legend()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
