import numpy as np
from Augmented_WCCI2020_numpy.Augmented_WCCI2020_numpy_dataset import Augmented_WCCI2020_Dataset



train_dataset, test_dataset = Augmented_WCCI2020_Dataset.get_datasets(dim=50,
                                              difficulty='easy')


for i in range(len(train_dataset)):
    tasks = train_dataset[i]  # 直接从 dataset 获取数据
    x = np.zeros(shape=(50,))
    print(tasks[0].eval(x))
    print(tasks[0].eval(x).dtype)
    print(tasks[0].eval(x).shape)
    print(type(tasks[0].eval(x)))
    # print(tasks[0].T1)




