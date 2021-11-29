import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn as sk

def remove_outlier(df, column):
    Q1=df.column.quantile(0.25)
    Q3=df.column.quantile(0.75)
    IQR=Q3-Q1
    lower_limit=Q1-1.5*IQR
    upper_limit=Q3+1.5*IQR
    return "hello"


if __name__ == '__main__':
    pd.set_option('display.max_columns', 10)
    data=pd.read_csv(r"C:/Users/harsh/Downloads/california_housing.csv")
    df=pd.DataFrame(data)
    df=df.drop(columns=['latitude','longitude'])
    prices=df['median_house_value']
    min_price=np.min(prices)
    max_price=np.max(prices)
    median_price=np.median(prices)
    mean_price=np.mean(prices)
    std_prices=np.std(prices)
    #sb.pairplot(df,size=1,plot_kws={'line_kws':{'color':'red'},'scatter_kws':{'alpha':0.1}},kind="reg")
    #sb.pairplot(df,hue='ocean_proximity')
    correlation_housing=df.corr()
    sb.heatmap(correlation_housing,annot=True)
    #plt.hist(data)
    #cm = np.corrcoef(data.T)
    #print(df)
    #print(type(data))
    #sb.heatmap(cm)
    print(df.describe())
    plt.show()



