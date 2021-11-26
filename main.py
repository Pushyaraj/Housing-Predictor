import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import sklearn as sk
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data=pd.read_csv(r"C:\Users\harsh\Downloads\housing.csv")
    prices=data['MEDV']
    min_price=np.min(prices)
    max_price=np.max(prices)
    median_price=np.median(prices)
    mean_price=np.mean(prices)
    std_prices=np.std(prices)
    sb.pairplot(data,size=2.5)
    #plt.hist(data)
    plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
