from statistics import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd

def polynomialRegression(dataframe, column, order):
    indexs_2 = np.array([i for i in range(len(dataframe[column].values))])
    indexs_2 = np.reshape(indexs_2, newshape=(1,-1))[0]

    RMSEs = []
    for i in range(order):
        features = PolynomialFeatures(degree=i+1)
        x_train_transformed = features.fit_transform(indexs_2.reshape(-1, 1))
        
        model = LinearRegression().fit(x_train_transformed, dataframe[column].values)

        train_pred = model.predict(x_train_transformed)
        rmse_poly_4_train = mean_squared_error(dataframe[column].values, train_pred, squared = False)
        RMSEs.append(rmse_poly_4_train)

    features = PolynomialFeatures(degree=RMSEs.index(min(RMSEs))+1)
    x_train_transformed = features.fit_transform(indexs_2.reshape(-1, 1))
    model = LinearRegression().fit(x_train_transformed, dataframe[column].values)

    x_test_transformed = features.fit_transform(indexs_2.reshape(-1, 1))

    train_pred = model.predict(x_train_transformed)

    test_pred = model.predict(x_test_transformed)
    
    dataframe[f'Poly{RMSEs.index(min(RMSEs))+1}'] = test_pred
    return test_pred


def fourrierFeatureGeneration(dataframe, column, order):
    close_fft = np.fft.rfft(np.asarray(dataframe[column].tolist()))
    fft_df = pd.DataFrame({'fft':close_fft})
    fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

    RMSEs_fft = []
    fft_list = np.asarray(fft_df['fft'].tolist())
    ffts=range(30)
    for num_ in ffts:
        num_ = num_+1
        fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_] = 0
        if len(dataframe) % 2 ==0:
            RMSEs_fft.append(mean_squared_error(dataframe[column].values, np.fft.irfft(fft_list_m10), squared = False))
        else:
            RMSEs_fft.append(mean_squared_error(dataframe[column].values, np.insert(np.fft.irfft(fft_list_m10),0,np.fft.irfft(fft_list_m10)[0],axis=0), squared = False))


    fft_list_m10= np.copy(fft_list)
    fft_list_m10[RMSEs_fft.index(min(RMSEs_fft))+1:-RMSEs_fft.index(min(RMSEs_fft))+1] = 0
    if len(dataframe) % 2 ==0:
        dataframe['fft{}'.format(RMSEs_fft.index(min(RMSEs_fft))+1)] = np.fft.irfft(fft_list_m10)
        return np.fft.irfft(fft_list_m10)
    else:
        dataframe['fft{}'.format(RMSEs_fft.index(min(RMSEs_fft))+1)] = np.insert(np.fft.irfft(fft_list_m10),0,np.fft.irfft(fft_list_m10)[0],axis=0)
        return np.insert(np.fft.irfft(fft_list_m10),0,np.fft.irfft(fft_list_m10)[0],axis=0)
        
        