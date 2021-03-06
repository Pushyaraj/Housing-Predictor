import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn as sk
from sklearn.model_selection import train_test_split
from matplotlib import interactive
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import preprocessing

def remove_outlier(df, column):
    #column="total_rooms"
    Q1=df[column].quantile(0.25)
    Q3=df[column].quantile(0.75)
    IQR=Q3-Q1
    lower_limit=Q1-1.7*IQR
    upper_limit=Q3+1.7*IQR
    df_outliers=df[~((df[column]<lower_limit)|(df[column]>upper_limit))]
    #print("IQR= {}, out of range bounds are: {},{}".format(IQR,lower_limit,upper_limit))
    #print("Outliers out of total = {} are \n {}".
          #format(data[column].size, len(df[column])))
    return df_outliers


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

    #checking for outliers

    """
    sb.boxplot(df['median_income'])
    plt.show()
    sb.boxplot(df['total_rooms'])
    plt.show()
    sb.boxplot(df['total_bedrooms'])
    plt.show()
    sb.boxplot(df['households'])
    plt.show()
    sb.boxplot(df['population'])
    plt.show()
    """


    #Removing outliers
    df=remove_outlier(df, "median_income")
    df=remove_outlier(df, "total_rooms")
    df=remove_outlier(df, "total_bedrooms")
    df=remove_outlier(df, "households")
    df=remove_outlier(df, "population")

    #print(data.shape)
    #print(df.shape)
    #data.boxplot(column='total_rooms')

    # Checking if outliers are removed
    """
    sb.boxplot(df['median_income'])
    plt.show()
    sb.boxplot(df['total_rooms'])
    plt.show()
    sb.boxplot(df['total_bedrooms'])
    plt.show()
    sb.boxplot(df['households'])
    plt.show()
    sb.boxplot(df['population'])
    plt.show()
    """
    #Succesfull

    #Removing null values
    # print(df.isnull().sum())
    # print(df)
    df.dropna(inplace=True)
    # print(df)

    #label encoding
    #label_encoder= preprocessing.LabelEncoder()
    #df['ocean_proximity'] = label_encoder.fit_transform(df['ocean_proximity'])
    #print(df['ocean_proximity'].value_counts())

    #one hot code encoding
    df=pd.get_dummies(df,columns=['ocean_proximity'])
    #print(df)

    # correlation heatmap
    """
    correlation_housing=df.corr()
    sb.heatmap(correlation_housing,annot=True)
    plt.show()
    """

    #Data that is cleaned and used for machine learning model
    #df.to_csv('clean_california_housing.csv')

    y = df['median_house_value']
    x = df.drop('median_house_value', axis=1)
    #print("{} {}".format(x.head(), y.head()))

    #sb.pairplot(df,x_vars=['median_income'], y_vars=['median_house_value'],kind='reg',
                #plot_kws={'scatter_kws':{'alpha':0.2},'line_kws':{'color':'black'}})
    train_x,test_x,train_y,test_y = train_test_split(x, y, test_size=0.2, random_state=1)  # Splitting the data

    #Testing the train/test data
    """
    print("train_x shape {} and size {}".format(train_x.shape, train_x.size))
    print("test_x shape {} and size {}".format(test_x.shape, test_x.size))
    print("train_y shape {} and size {}".format(train_y.shape, train_y.size))
    print("test_y shape {} and size {}".format(test_y.shape, test_y.size))
    """


    scaler = StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)


    lr = LinearRegression()
    lr.fit(train_x,train_y)
    #y_pred=lr.predict(standardized_X_test)
    #print("ahss {}   {}".format(y_pred,train_y))
    #r2=r2_score(y_true=train_y,y_pred=y_pred)
    print("Intercept is "+str(lr.intercept_))
    y_pred=lr.predict(test_x)
    print(len(y_pred))
    print(len(test_y))

    testing=pd.DataFrame({'Actual':test_y, 'Predicted':y_pred})
    sb.jointplot(x='Actual', y='Predicted', data=testing, kind='reg',color='r'
                 ,joint_kws={'line_kws':{'color':'black'}} )

    print(np.sqrt(mean_squared_error(test_y, y_pred)))
    print(np.sqrt(mean_squared_error(train_y, lr.predict(train_x))))

    #plt.scatter(test_x,test_y)
    sb.histplot(data['median_income'])
    plt.show()
    #sb.boxplot(df['total_rooms'])
    sb.histplot(df['median_income'])
    plt.show()

    #df.boxplot(column='population')
    #plt.hist(data)
    #cm = np.corrcoef(data.T)
    #print(df)
    #print(type(data))
    #sb.heatmap(cm)
    #print(df.describe())
    interactive(False)
    plt.show()





