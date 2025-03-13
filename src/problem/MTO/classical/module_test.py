import numpy as np
# from WCCI2020_numpy.WCCI2020_numpy_dataset import WCCI2020_Dataset
# from CEC2017_numpy.CEC2017_numpy_dataset import CEC2017_Dataset
from WCCI2020_torch.WCCI2020_torch_dataset import WCCI2020_Dataset
from CEC2017_torch.CEC2017_torch_dataset import CEC2017_Dataset


#dataset = WCCI2020_Dataset()
dataset = CEC2017_Dataset()

for i in range(len(dataset)):
    tasks = dataset[i]  # 直接从 dataset 获取数据
    x = np.zeros(shape=(2,50))
    print(tasks[0].eval(x)[0],tasks[0].eval(x)[1])
    print(tasks[0].eval(x)[0].dtype,tasks[1].eval(x)[1].dtype)
    print(type(tasks[0].eval(x)[0]),type(tasks[1].eval(x)[1]))
    print(tasks[0].T1)




