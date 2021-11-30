import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn as sk

def remove_outlier(df, column):
    #column="total_rooms"
    Q1=df.total_rooms.quantile(0.25)
    Q3=df.total_rooms.quantile(0.75)
    IQR=Q3-Q1
    lower_limit=Q1-1.5*IQR
    upper_limit=Q3+1.5*IQR
    print("IQR= {}, out of range bounds are: {},{}".format(IQR,lower_limit,upper_limit))
    #rdf=df[df['total_rooms']<lower_limit|df['total_rooms']>upper_limit]
    lower=np.where(df['total_rooms']<=lower_limit)
    upper=np.where(df['total_rooms']>=upper_limit)
    df.drop(lower[0], inplace=True)
    df.drop(upper[0], inplace=True)

    #removed_outlier=df.drop(rdf)
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

    #correlation heatmap
    correlation_housing=df.corr()
    sb.heatmap(correlation_housing,annot=True)

    #Removing outlier
    df_outlier=remove_outlier(df,"total_rooms")
    print(data.shape)
    print(df.shape)

    #plt.hist(data)
    #cm = np.corrcoef(data.T)
    #print(df)
    #print(type(data))
    #sb.heatmap(cm)
    print(df.describe())
    #plt.show()



