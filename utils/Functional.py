import os
import numpy as np
from h5py.h5t import cfg
from matplotlib import pyplot as plt
from prettytable import PrettyTable
import pandas as pd
import xlrd


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list, device: str, normalize=True):
        self.matrix = np.zeros((num_classes, num_classes), dtype="float32")
        self.num_classes = num_classes
        self.labels = labels
        self.device = device
        self.normalize = normalize

    def update(self, preds, labels):
        for t, p in zip(labels, preds):
            self.matrix[t, p] += 1

    def getMatrix(self, normalize=True):
        """
        if normalize=True, matrix is in percentage form
        if normalize=False, matrix is in numerical  form
        """
        if normalize:
            per_sum = self.matrix.sum(axis=1)
            for i in range(self.num_classes):
                self.matrix[i] = (self.matrix[i] / per_sum[i])
                self.matrix = np.around(self.matrix, 2)
                self.matrix[np.isnan(self.matrix)] = 0
        return self.matrix

    def plot(self):
        matrix = self.getMatrix(self.normalize)
        # print("---------Confusion Matrix--------")
        # print(matrix)
        plt.figure(figsize=(15, 15))
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=90, fontsize=12)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels, fontsize=12)
        # 显示colorbar
        plt.colorbar(fraction=0.046, extend='both')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = float(format('%.2f' % matrix[y, x]))
                plt.text(x, y, info,
                         fontsize=10,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()  # 保证图不重叠
        plt.savefig('plot/' + self.device + '_confusion_matrix.svg', format='svg')
        plt.show()

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)
        print('\n')

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "F1_score"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FN = np.sum(self.matrix[i, :]) - TP
            FP = np.sum(self.matrix[:, i]) - TP
            # TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            F1_score = round(2 * Precision * Recall / (Precision + Recall), 3) if Precision + Recall != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, F1_score])
        print(table)


class Table(object):
    def __init__(self, title, index, columns, sheet_name):
        self.title = title
        self.index = index
        self.columns = columns
        self.sheet_name = sheet_name
        self.index_len = len(index)
        self.columns_len = len(columns)
        self.header = np.array([columns], dtype=str)
        self.header[0] = self.title
        self.data = np.zeros(shape=(self.index_len, self.columns_len))
        self.df = None

    def add(self, row, col, data):
        index_row = np.where(np.array(self.index) == row)
        if type(col) == str:
            index_col = np.where(np.array(self.columns) == col)
        elif type(col) == int:
            index_col = col
        else:
            index_col = 0
        self.data[index_row, index_col] = data

    def save_rect(self, path):
        file_path = path + '/result_table.xlsx'
        # if self.columns_len > 1:
            # average = (self.data[:, 0] + self.data[:, 1]) / 2.
            # self.data[:, 3] = average
        df = pd.DataFrame(self.data, index=self.index, columns=self.columns)
        # df = pd.DataFrame(self.data, index=self.index, columns=['aug/na', 'na/aug', 'aug/aug', 'average'])

        df_title = pd.DataFrame([], index=[self.title])
        # print(np.array(self.columns).reshape(1, -1))
        df_head = pd.DataFrame(np.array(self.columns).reshape(
            1, -1), index=['aug'], columns=self.columns)

        df = pd.concat([df_title, df_head, df])
        if os.path.exists(file_path):
            df_old = pd.read_excel(file_path, index_col=0, sheet_name=self.sheet_name)
            df = pd.concat([df_old, df])
        df.to_excel(file_path, sheet_name=self.sheet_name)

        return file_path

    def save_square(self, path):
        file_path = path + '/result_table.xlsx'
        df = pd.DataFrame(self.data, index=self.index, columns=self.columns)
        # df = pd.DataFrame(self.data, index=self.index, columns=['aug/na', 'na/aug', 'aug/aug', 'average'])

        df_title = pd.DataFrame([], index=[self.title])
        # print(np.array(self.columns).reshape(1, -1))
        df_head = pd.DataFrame(np.array(self.columns).reshape(
            1, -1), index=['aug1/aug2'], columns=self.columns)

        df_title_2 = pd.DataFrame([], index=['average of one view'])
        average = (self.data[0, :] + self.data[:, 0]) / 2.
        df_2 = pd.DataFrame([average], index=['average'], columns=self.columns)

        df = pd.concat([df_title, df_head, df, df_title_2, df_2])
        if os.path.exists(file_path):
            df_old = pd.read_excel(file_path, index_col=0, sheet_name=self.sheet_name)
            df = pd.concat([df_old, df])
        df.to_excel(file_path, sheet_name=self.sheet_name)

        return file_path

