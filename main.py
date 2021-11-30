import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn as sk
from sklearn.model_selection import train_test_split
from matplotlib import interactive
from sklearn import preprocessing

def remove_outlier(df, column):
    #column="total_rooms"
    Q1=df[column].quantile(0.25)
    Q3=df[column].quantile(0.75)
    IQR=Q3-Q1
    lower_limit=Q1-1.6*IQR
    upper_limit=Q3+1.6*IQR
    #print("IQR= {}, out of range bounds are: {},{}".format(IQR,lower_limit,upper_limit))
    #rdf=df[df['total_rooms']<lower_limit|df['total_rooms']>upper_limit]
    lower=np.where(df[column]<=lower_limit)
    upper=np.where(df[column]>=upper_limit)
    df.drop(lower[0], inplace=True)
    df.drop(upper[0], inplace=True)

    #removed_outlier=df.drop(rdf)
    return 0


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
    #correlation_housing=df.corr()
    #sb.heatmap(correlation_housing,annot=True)

    #Removing outlier
    df_outlier=remove_outlier(df, "median_income")
    #print(data.shape)
    #print(df.shape)
    #data.boxplot(column='total_rooms')

    # print(df.isnull().sum())
    # print(df)
    df.dropna(inplace=True)
    # print(df)
    
    label_encoder= preprocessing.LabelEncoder()
    df['ocean_proximity'] = label_encoder.fit_transform(df['ocean_proximity'])
    #print(df['ocean_proximity'].value_counts())



    y = df['median_house_value']
    x = df['median_income']

    #sb.pairplot(df,x_vars=['median_income'], y_vars=['median_house_value'],kind='reg',
                #plot_kws={'scatter_kws':{'alpha':0.2},'line_kws':{'color':'black'}})
    #train_x=x[:80]
    #train_y=y[:80]
    #test_x=x[80:]
    #test_y=y[80:]
    train_x,train_y,test_x,test_y=train_test_split(x,y,test_size=0.2)  # Splitting the data

    print("X_train shape {} and size {}".format(train_x.shape, train_x.size))
    print("X_test shape {} and size {}".format(test_x.shape, test_x.size))
    print("y_train shape {} and size {}".format(train_y.shape, train_y.size))
    print("y_test shape {} and size {}".format(test_y.shape, test_y.size))
    print("\n {} \n {}".format(train_x,train_y))
    #plt.scatter(test_x,test_y)
    sb.histplot(data['median_income'])
    plt.show()
    sb.histplot(df['median_income'])


    #df.boxplot(column='population')
    #plt.hist(data)
    #cm = np.corrcoef(data.T)
    #print(df)
    #print(type(data))
    #sb.heatmap(cm)
    #print(df.describe())
    interactive(False)
    plt.show()





