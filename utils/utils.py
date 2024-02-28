from scipy.signal import stft
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#function def
#normalization
def norml(x):
  siz = x.shape
  xn = x
  for i in range(siz[0]):
    for j in range(3):
      tem = x[i,:,j]
      maxx = np.max(abs(tem))
      minn = np.min(tem)
      meann = np.mean(tem)
      xn[i,:,j] = (tem-meann)/maxx
      
  return xn

#muti-channel batch STFT

def batch_stft(x, windowf='boxcar', nfft=256, overlap=0.9):
  siz = x.shape
  num_channels = x.shape[2]  # Number of channels
  f, t, Zxx = stft(x[0, :, 0], window = windowf, nperseg=nfft, noverlap=int(nfft*overlap))  # Perform STFT on a single channel to get the dimensions
  num_frequencies = len(f)
  num_time_segments = len(t)

  xf = np.zeros((siz[0], num_frequencies, num_time_segments, num_channels))
  for i in range(siz[0]):
    for j in range(num_channels):
      tem = x[i, :, j]
      f, t, Zxx = stft(tem, window = windowf, nperseg=nfft, noverlap=int(nfft*overlap))
      maxx = np.max(abs(Zxx))
      xf[i, :, :, j] = abs(Zxx)

  return xf

def resultshow(classes,label):
  lis=np.zeros(classes.shape[0])
  for i in range(classes.shape[0]):
   jeg = np.where(classes[i,:] == max(classes[i,:]))
   lis[i] = jeg[0]

  tru=np.zeros(classes.shape[0])
  for i in range(classes.shape[0]):
    jeg2 = np.where(label[i,:] == max(label[i,:]))
    tru[i] = jeg2[0]
  
  target = ['quake','earthquake','rockfall','enviroemnt noise']
  print(classification_report(tru, lis, target_names=target))
  print(confusion_matrix(tru, lis))
  
  
  #sliding window
def streamread(daydata):
  step =250
  num_window = int(daydata.shape[1]/step)
  transfer_set = np.zeros((num_window,3750,3),dtype="float32")
  for i in range(num_window):
    start = i*step;
    eend = start+3750;
    if eend > daydata.shape[1]:
     break
    transfer_set[i,:,:] = daydata[0,start:eend,:]
  return transfer_set



#post-processing
from scipy.signal import gaussian
from scipy.ndimage import median_filter

def smooth_softmax_results(softmax_results, threshold=0.7, med_filter_size=5, kernel_size=15, kernel_sigma=4):
    # Apply post-processing threshold to input results
    possibility = softmax_results.copy()
    possibility[possibility < threshold] = 0

    # Apply median filter to each class separately
    filtered_possibility = np.zeros_like(possibility)
    for i in range(possibility.shape[1]):
        filtered_possibility[:, i] = median_filter(possibility[:, i], size=med_filter_size)

    # Check if filtered_possibility is all zeros and return zeros in that case
    if not np.any(filtered_possibility):
        return np.zeros_like(filtered_possibility)

    # Define the filter kernel with a sum of 1 and length 'kernel_size'
    kernel = gaussian(kernel_size, kernel_sigma)
    kernel /= np.sum(kernel)

    # Pad the filtered possibility with zeros to handle edge cases
    filtered_possibility_padded = np.pad(filtered_possibility, ((kernel_size // 2, kernel_size // 2), (0, 0)), mode='constant')

    # Apply the convolution operation separately to each class
    smoothed_results = []
    for c in range(filtered_possibility.shape[1]):
        smoothed_c = np.convolve(filtered_possibility_padded[:, c], kernel, mode='valid')
        #smoothed_c = (smoothed_c - np.min(smoothed_c)) / (np.max(smoothed_c) - np.min(smoothed_c))
        smoothed_results.append(smoothed_c)

    # Stack the smoothed results for each class into a single array
    smoothed_results = np.stack(smoothed_results, axis=1)

    return smoothed_results
   
 
import numpy as np
def find_misclassified_events(predicted_probs, true_probs):
    """
    Given predicted probabilities and true probabilities, return a dictionary containing information about the misclassified events grouped by the type of error.
    """
    predicted_labels = np.argmax(predicted_probs, axis=1) # get the index of the max probability for each event in predicted probabilities
    true_labels = np.argmax(true_probs, axis=1) # get the index of the max probability for each event in true probabilities
    misclassified_indices = np.flatnonzero(predicted_labels != true_labels)
    misclassified_info = {} # dictionary to store misclassification information
    for idx in misclassified_indices:
        predicted_label = predicted_labels[idx]
        true_label = true_labels[idx]
        error_type = f'predicted_label_{predicted_label}_true_label_{true_label}'
        if error_type not in misclassified_info:
            misclassified_info[error_type] = {'indices': [idx]}
        else:
            misclassified_info[error_type]['indices'].append(idx)
    match_indices = np.flatnonzero(predicted_labels == true_labels)
    return {'misclassified_info': misclassified_info, 'match_indices': match_indices}

def print_misclassified_info(misclassified_info):
    label_map = {0: 'Quake', 1: 'Earthquake', 2: 'Rockfall', 3: 'Noise'}
    for error_type, info in misclassified_info.items():
        predicted_label = int(error_type.split('predicted_label_')[1].split('_true_label_')[0])
        true_label = int(error_type.split('_true_label_')[1])
        predicted_name = label_map[predicted_label]
        true_name = label_map[true_label]
        print(f"{true_name} predicted as {predicted_name}:")
        print(f"Indices: {info['indices']}")
        print(f"Number of misclassifications: {len(info['indices'])}")
        print()
