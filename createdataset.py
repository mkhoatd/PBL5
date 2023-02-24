# %%
import os
import warnings
import zipfile
import matplotlib.pyplot as plt
import scipy.io
import biosppy
import cv2
import numpy as np

warnings.filterwarnings("ignore")
os.environ["QT_QPA_PLATFORM"] = "offscreen"


file_name = "ECG signals (1000 fragments).zip"
zip_file_path = os.path.join("./", file_name)
unzip_file_path = os.path.join("./", "MLII")
if not os.path.exists(unzip_file_path):
    # Unzip the file if not unzipped
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall("./")
# %%


class SignalToImage:
    def __init__(self, dataset_path, output_path):
        """
        dataset_path: str, path to the dataset
        output_path: str, path to the output directory
        """
        self.dataset_path = dataset_path
        self.output_path = output_path
        if not os.path.exists(output_path):
            # if output directory not exist, create it
            os.makedirs(output_path)
        self.preprocess_dataset()

    # Phân đoạn array
    def segmentation(self, signal: np.ndarray) -> list[np.ndarray]:
        """signal: np.ndarray
        segment signal into multiple signals each contain a heart beat
        return: list of np.ndarray
        """
        data = signal
        signals = []
        count = 1
        peaks = biosppy.signals.ecg.christov_segmenter(
                                                        signal=data, 
                                                        sampling_rate=360)[0]
        for i in peaks[1:-1]:
            # Khoang cach giua dinh nay va dinh truoc -20
            diff1 = abs(peaks[count - 1] - i) - 20
            # Khoang cach giua dinh nay va dinh sau -20
            diff2 = abs(peaks[count + 1] - i) - 20
            # centering the peak
            signal_range = min(diff1, diff2)
            x = peaks[count] - signal_range
            y = peaks[count] + signal_range
            signal = data[x:y]
            signals.append(signal)
            count += 1
        return signals

    # Convert signal to image
    def signal_to_img(self, signal: np.ndarray, filename: str) -> None:
        """
        signal: np.ndarray
        filename: str
        """
        fig = plt.figure(frameon=False)
        plt.plot(signal)
        plt.xticks([]), plt.yticks([])
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        fig.savefig(filename)
        im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        im_gray = cv2.resize(im_gray, (128, 128), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(filename, im_gray)
        plt.cla()
        plt.clf()
        plt.close("all")
        # self.array_to_img(array, self.output_directory)

    def preprocess_dataset(self) -> None:
        folders = os.listdir(self.dataset_path)
        if len(folders[0].split()) == 1:
            return
        for folder in folders:
            name = folder.split()[1]
            os.rename(
                os.path.join(self.dataset_path, folder),
                os.path.join(self.dataset_path, name),
            )

    def convert(self) -> None:
        """
        Convert dataset into form useable in tensorflow"""
        ecg_classes = os.listdir(self.dataset_path)
        count = 0
        for ecg_class in ecg_classes:
            ecg_class_path = os.path.join(self.dataset_path, ecg_class)
            ecg_files = os.listdir(ecg_class_path)
            signals = []
            for ecg_file in ecg_files:
                ecg_file_path = os.path.join(ecg_class_path, ecg_file)
                mat = scipy.io.loadmat(ecg_file_path)
                values = mat.get("val")[0]
                # signals[count].append(self.segmentation(values))
                signals += self.segmentation(values)
            count = 1
            for signal in signals:
                directory = os.path.join(self.output_path, ecg_class)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = os.path.join(
                    self.output_path, ecg_class, f"{ecg_class}_original_image_{count}.jpg"
                )
                self.signal_to_img(signal, filename)
                count += 1


# %%

a = SignalToImage("MLII", "Dataset")
a.convert()

# %%
