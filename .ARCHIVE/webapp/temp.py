from sodapy import Socrata
from os.path import isfile
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns



# # fetch the latest parkign data and append to parking sensor file collection
# def update_latest_parking_data():
#     # connect to melbourne open data apis
#     client = Socrata(
#         "data.melbourne.vic.gov.au",
#         "EC65cHicC3xqFXHHvAUICVXEr", # app token, just used to reduce throttling, not authentication
#         timeout=120
#     )
    
#     # add another number to the csv
#     df = pd.DataFrame(client.get("vh2v-4nfs", limit=200000))
#     df.drop(columns = ['location', 'lat', 'lon', ':@computed_region_evbi_jbp8'], inplace = True)
#     df['datetime'] = datetime.today().replace(microsecond=0) + datetime.timedelta(hours = 10) #Melbourne Time
    
#     # if file already exists than we just want to append to the file
#     # else we want to write a new file
#     write_mode = 'w' if isfile('parking_sensor_data.csv') else 'a'
    
#     df.to_csv('parking_sensor_data.csv', mode=write_mode, index = False)

# import time

# # simply press keys `ctrl + c` to stop the loop
# try:
#     while True:
#         update_latest_parking_data()
#         # wait for 15 minutes before fetching next latest
#         time.sleep(20)
# except KeyboardInterrupt:
#     print('Finished fetching parking data.')



# load data from csv file
df = pd.read_csv('parkingsensor.csv', parse_dates=['datetime'])
# add 'DayOfWeek' column
df['DayOfWeek'] = df['datetime'].dt.day_name()

# subset of data for only todays date
current_df = df[df['datetime'].dt.date == datetime.now().date()]

'''
This function will take in a data frame with entries for each sensor with respective day of week, and status columns
and return a dataframe of the form [{'DayOfWeek': string, 'Percentage': float32}]
'''
def get_daily_percentage_availability(df):
    DailyParkingCounts = df.groupby('DayOfWeek').status.value_counts()
    DailyParkingCounts = DailyParkingCounts.unstack().reset_index()
    DailyParkingCounts['Percentage'] = (DailyParkingCounts['Unoccupied'] / (DailyParkingCounts['Unoccupied'] + DailyParkingCounts['Present']))
    DailyParkingCounts.reset_index(drop=True)
    return DailyParkingCounts


def get_hourly_availability_trend(df):
    df['Hours'] = df['datetime'].dt.hour
    DailyAvailability = df.groupby('Hours').status.value_counts()
    DailyAvailability = DailyAvailability.unstack().reset_index()
    DailyAvailability['Availability'] = DailyAvailability['Unoccupied'] / (DailyAvailability['Present'] + DailyAvailability['Unoccupied'])
    DailyAvailability = DailyAvailability.reset_index(drop=True)
    return DailyAvailability[['Hours', 'Availability']]


'''
    This function takes in an expected daily trend DataFrame as produced by the function above,
    and also a smaller 'current' DataFrame which has the same schema but with data only covering the current day
    of parking sensor activity.
    
    The output will be a visualization of the expected and current trends
'''
def visualize_trend(expected, current, x_column='DayOfWeek', y_column = 'Percentage'):
    # Visualize the results
    sns.set(font_scale=1.5)
    plt.figure(figsize=(12, 6), dpi=80)
    sns.set_style("whitegrid")

    plt.ylabel("% Available", labelpad=14)
    plt.title("Parking Availability", y=1)

    plt.bar(expected[x_column], expected[y_column], alpha=0.4 , label="Expected")
    plt.bar(current[x_column], current[y_column], alpha=0.4 , label="Current")
    # plt.bar(WednesdayCount['Day_Of_Week'], WednesdayCount['Parking_Availabilities'],alpha=0.4, label="Available Now")
    plt.legend(loc ="lower left", borderaxespad=1)
    plt.show()

# perform analysis
daily_percentage = get_daily_percentage_availability(df)
# perform analysis limited to today
current_daily_percentage = get_daily_percentage_availability(current_df)
#visualize results
visualize_trend(daily_percentage, current_daily_percentage)


# def get_hourly_availability_trend(df):
#     df['Hours'] = df['datetime'].dt.hour


    