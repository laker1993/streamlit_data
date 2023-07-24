import streamlit as st
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt

# 展示文本；文本直接使用Markdown语法
st.markdown("# LTV计算器demo")
st.markdown("""
            #### V1版本
            """)


# 加入交互控件，如输入框
sec_ret = st.number_input("次日留存为",0.2)
third_ret = st.number_input("3日留存为",0.15)
four_ret = st.number_input("4日留存为",0.13)
five_ret = st.number_input("5日留存为",0.12)
need_days = st.number_input("预估天数为",15)
# six_ret = st.number_input("6日留存为", 0.235)
# seven_ret = st.number_input("7日留存为", 0.228)

arpu = st.number_input("APRU为",0.0000001, format="%0.7f")

rat=[sec_ret,third_ret,four_ret,five_ret]

# scipy的curve_fit函数进行曲线拟合,模型为指数衰减模型
# pkg和day为1-3的数据作为已知xdata和ydata,进行曲线拟合,然后基于得到的模型参数预测1-30天的留存率
from optimize import curve_fit
import numpy as np

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

st.write("LT预估为：", np.sum(rat_predict))
st.write("LTV预估为：", np.sum(rat_predict)*arpu)


# x = range(0,need_days-1)
# y2 = list(rat_predict)

# plt.figure(figsize=(10,5))
# plt.plot(x, y2, color='pink', marker='^', linewidth=2, label='predict')

# plt.title('rat_predict')
# plt.xlabel('days')  
# plt.ylabel('rat')
# plt.legend()
# st.set_option('deprecation.showPyplotGlobalUse', False)
# st.pyplot()

# 怎么在github 使用 matplotlib
