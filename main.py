import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import sklearn as sk
# Press the green button in the gutter to run the script.
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
    #sb.pairplot(df,size=1)
    #plt.hist(data)
    sb.regplot(x='median_income',y='median_house_value',data=df,scatter_kws={'alpha':0.05})
    #cm = np.corrcoef(data.T)
    #print(df)
    #print(type(data))
    #sb.heatmap(cm)
    print(df.describe())
    plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
