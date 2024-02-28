import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np

def get_event_time(event_num, file_path):
    df = pd.read_excel(file_path)
    event_row = df.loc[df.iloc[:,0] == event_num].iloc[0]
    year = event_row[1]
    month = event_row[2]
    day = event_row[3]
    hour = event_row[4]
    mins = event_row[5]
    sec = event_row[6]
    event_class = event_row[7]
    second_of_day = (hour * 3600) + (mins * 60) + sec
    return day, second_of_day

def get_event_info(event_num, file_path = './data/label25to28.xlsx'):
    df = pd.read_excel(file_path)
    event_row = df.loc[df.iloc[:,0] == event_num].iloc[0]
    year = event_row[1]
    month = event_row[2]
    day = event_row[3]
    hour = event_row[4]
    mins = event_row[5]
    sec = event_row[6]
    event_class = event_row[7]
    second_of_day = (hour * 3600) + (mins * 60) + sec
    return day, hour, mins, sec, event_class

def get_events_on_day(day, file_path):
    df = pd.read_excel(file_path)
    day_rows = df.loc[df.iloc[:,3] == day]
    events_on_day = []
    event_classes = []
    for i, row in day_rows.iterrows():
        hour = row[4]
        mins = row[5]
        sec = row[6]
        second_of_day = (hour * 3600) + (mins * 60) + sec
        events_on_day.append(second_of_day)
        event_classes.append(row[7])
    return events_on_day, event_classes


def get_final_category(input):
    """
    Takes in the post-processed output probabilities from a CNN model with
    shape [time_step, classes], and returns a one-hot encoded numpy array
    representing the final category based on the highest probability over
    the relevant categories.
    """
    # extract the probabilities for the relevant categories (quake, earthquake, rockfall)
    relevant_probs = input[:, :3]

    # check if the maximum probability for any of the relevant categories is greater than 0.2
    if np.max(relevant_probs) > 0.18:
        # if so, choose the category with the highest probability as the final result        
        max_index = np.argmax(relevant_probs)
        max_time_index, final_category_index = np.unravel_index(max_index, relevant_probs.shape)
    else:
        # otherwise, set the final result to "no-event noise"
        final_category_index = 3

    # create a one-hot encoded numpy array representing the final category
    final_category = np.zeros((1, 4))
    final_category[0, final_category_index] = 1

    return final_category

def get_softmax_stream_slice(number, full_stream):
    return full_stream[number-10:number+20, :]

#plot softmax reslts stream
def plot_possibility_series(possibility_series, start_index, end_index):
    plt.figure(figsize=(5, 3))
    plt.plot(possibility_series[start_index:end_index, 0], label='quake')
    plt.plot(possibility_series[start_index:end_index, 1], label='earthquake')
    plt.plot(possibility_series[start_index:end_index, 2], label='rockfall')
    plt.legend(loc='upper right')
    plt.show()



