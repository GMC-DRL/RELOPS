from kan import *
from mlp import MLP
from .basic_problem import Basic_Problem
from problem.SOO.bbob_numpy.bbob import *

from torch.utils.data import Dataset

class bbob_surrogate_model(Basic_Problem):
	def __init__(self, dim, func_id, lb, ub, shift, rotate, bias, difficulty, config):
		self.dim = dim
		self.func_id = func_id
		# shift = np.zeros(dim)
		# bias = 0
		# rotate = np.eye(dim)
		self.instance = eval(f'F{func_id}')(dim=dim, shift=shift, rotate=rotate, bias=bias, lb=lb, ub=ub)
		self.device = config.device
		self.optimum = None


		if func_id in [1, 3, 4, 6, 7, 10, 12, 13, 14, 15, 23, 24]:
			self.model = KAN.loadckpt(f'./problem/data_files/surrogate_model/Dim{dim}/KAN/{self.instance}/model')
		elif func_id in [2, 5, 8, 9, 11, 16, 17, 18, 19, 20, 21, 22]:
			self.model = MLP(dim)
			self.model.load_state_dict(
				torch.load(f'./problem/data_files/surrogate_model/Dim{dim}/MLP/{self.instance}/model.pth')
			)
		self.model.to(self.device)
		# KAN: 1,3,4,6,7,10,12,13,14,15,23,24  MLP:2,5,8,9,11,16,17,18,19,20,21,22

		self.ub = ub
		self.lb = lb

	def eval(self, x):
		# if is_train:
		# 	input_x = (x - self.lb) / (self.ub - self.lb)
		# 	with torch.no_grad():
		# 		y = self.model(input_x)
		# 	return y
		# else:

		# 	return self.instance.eval(x)
		if isinstance(x, np.ndarray):
			x = torch.tenor(x).to(self.device)
		input_x = (x - self.lb) / (self.ub - self.lb)
		with torch.no_grad():
			y = self.model(input_x)
		return y.detach().numpy()
		# return y

	def __str__(self):
		return f'Surrogate_{self.instance}'


class bbob_surrogate_Dataset(Dataset):
	def __init__(self,
				 data,
				 batch_size=1):
		super().__init__()
		self.data = data
		self.batch_size = batch_size
		self.N = len(self.data)
		self.ptr = [i for i in range(0, self.N, batch_size)]
		self.index = np.arange(self.N)

	@staticmethod
	def get_datasets(dim, config, upperbound=5, train_batch_size=1,
					 test_batch_size=1, difficulty='easy',is_train=True,seed=3849,
					 shifted=False, biased=False, rotated=False):
		
		# train_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
		# 			20]
		# test_id = [16, 17, 18, 19, 21, 22, 23, 24]

		if difficulty == 'easy':
			train_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
					20]
			# test_id = [16, 17, 18, 19, 21, 22, 23, 24]
		elif difficulty == 'difficult':
			train_id = [1, 2, 4, 5, 6, 10, 11, 13]
		else:
			raise ValueError(f'{difficulty} difficulty is invalid.')

		np.random.seed(seed)
		train_set = []
		test_set = []
		ub = upperbound
		lb = -upperbound

		# shift = torch.zeros(dim, dtype=torch.float64)
		# bias = 0.0
		# rotate = torch.eye(dim, dtype=torch.float64)
		# shift = np.zeros(dim)
		# bias = 0.0
		# rotate = np.eye(dim)
		func_id = [i for i in range(1, 25)]
		for id in func_id:
			if shifted:
				shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
			else:
				shift = np.zeros(dim)
			if rotated:
				H = rotate_gen(dim)
			else:
				H = np.eye(dim)
			if biased:
				bias = np.random.randint(1, 26) * 100
			else:
				bias = 0
			# instance = eval(f'F{id}')(dim=dim, shift=shift, rotate=H, bias=bias, lb=lb, ub=ub)
			# print(bias)

			if id in train_id:
			
				if is_train:
					train_instance = bbob_surrogate_model(dim, id, ub=ub, lb=lb, shift=shift, rotate=H, bias=bias,
												difficulty=difficulty,config=config)
				else:
					train_instance = eval(f'F{id}')(dim=dim, shift=shift, rotate=H, bias=bias, lb=lb, ub=ub)
				
				train_set.append(train_instance)

			# if id in test_id:
			else:
				test_instance = eval(f'F{id}')(dim=dim, shift=shift, rotate=H, bias=bias, lb=lb, ub=ub)
				test_set.append(test_instance)

		return bbob_surrogate_Dataset(train_set, train_batch_size), bbob_surrogate_Dataset(test_set, test_batch_size)

	def __len__(self):
		return self.N

	def __getitem__(self, item):

		if self.batch_size < 2:
			return self.data[self.index[item]]
		ptr = self.ptr[item]
		index = self.index[ptr: min(ptr + self.batch_size, self.N)]
		res = []
		for i in range(len(index)):
			res.append(self.data[index[i]])
		return res

	def __add__(self, other: 'bbob_surrogate_Dataset'):
		return bbob_surrogate_Dataset(self.data + other.data, self.batch_size)

	def shuffle(self):
		self.index = torch.randperm(self.N)
