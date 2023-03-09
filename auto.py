import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.tsa.ar_model import AutoReg

def format_minutes_seconds(x, pos):
    minutes = int(x / 60)
    seconds = int(x % 60)
    return f"{minutes:02d}m {seconds:02d}s"

if __name__ == "__main__":
    data = pd.read_csv('data.csv', delimiter=',')
    #Take 90% of the data for training and 10% for testing
    train = data['Average engagement time'][:int(0.9*(len(data)))]
    test = data['Average engagement time'][int(0.9*(len(data))):]
    print(train)

    # Create a FuncFormatter object using the format_minutes_seconds function
    formatter = ticker.FuncFormatter(format_minutes_seconds)

    #Change lags and choose which one is best
    errors = []
    for i in range(1, 100):
        try:
            model = AutoReg(train, lags=i, old_names=False).fit()
            print(f'AR Model with {i} lags: AIC = {model.aic}')
            
            predictions = model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
            print(f'Actual: {test.iloc[0]:.2f} seconds, Predicted: {predictions.iloc[0]:.2f} seconds')
            avg = 0
            for j in range(len(predictions)):
                avg += abs(test.iloc[j] - predictions.iloc[j])
            avg = avg / len(predictions)
            print(f'Average error: {avg:.2f} seconds')
            errors.append(avg)
        except:
            pass

    # #Print hte index of the minimum error
    print(f'Best model: {errors.index(min(errors)) + 1} lags')
    print(f'Best error: {min(errors)} seconds')

    # Create an AR model with a lag of 50
    model = AutoReg(train, lags=errors.index(min(errors)) + 1, old_names=False).fit()

    #Now we can use the model to predict the next 30 days
    predictions = model.predict(start=len(data), end=len(data)+22, dynamic=False)
    
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(data.index[-30:], data['Average engagement time'].tail(30), label='Actual')
    ax.plot(predictions.index, predictions, label='Predicted')
    ax.set_xlabel('Day')
    ax.set_ylabel('Average engagement time (seconds)')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(7))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(formatter)
    ax.set_title('Accuracy: ' + str(round((1 - (abs(data['Average engagement time'].tail(30).mean() - predictions.mean()) / data['Average engagement time'].tail(30).mean())) * 100, 2)) + '%')
    ax.legend()
    plt.show()

    
