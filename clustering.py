from numpy import *
from scipy import linalg,optimize

class User():
	def __init__(self):
		self.num_of_movies = 3952
		self.num_of_users = 6040
		self.num_of_features = 16
		self.lambda_value = 5
		self.MAX_ITERS = 10
		self.Y = asmatrix(zeros((self.num_of_movies,self.num_of_users)))
		self.loadRatingsFileToArray('ml-1m/ratings.dat')
		self.R = (self.Y != 0).astype(int)
		self.X = asmatrix(random.randn(self.num_of_movies,self.num_of_features))
		self.Theta = asmatrix(random.randn(self.num_of_users,self.num_of_features))
		self.predict_movie()
	def loadRatingsFileToArray(self,filename):
		f = open(filename)
		for line in f.readlines():
			ur = line.strip().split('::')
			self.Y[int(ur[1])-1,int(ur[0])-1] = ur[2]
		f.close()
	def costFunction(self,params):
		self.callback(params)
		return (1/2 * sum(multiply(power(self.X.dot(transpose(self.Theta)) - self.Y,2),self.R))) + ( (self.lambda_value/2)* (sum(power(self.Theta,2)) + sum(power(self.X,2))))
	def costFunctionGrad(self,params):
		self.callback(params)
		X_grad = (multiply(self.X.dot(transpose(self.Theta)) - self.Y,self.R).dot(self.Theta)) + (self.lambda_value * self.X)
		Theta_grad = (transpose(multiply(self.X.dot(transpose(self.Theta)) - self.Y,self.R)).dot(self.X)) + (self.lambda_value * self.Theta)
		return ravel(vstack((X_grad,Theta_grad)))
	def normalizeY(self):
		m,n = shape(self.Y)
		Y_mean = asmatrix(zeros((m, 1)))
		Y_norm = asmatrix(zeros((m,n)))
		for i in range(m):
			idx = self.R[i,:].nonzero()[1]
			Y_mean[i] = mean(self.Y[i,idx])
			Y_norm[i,idx] = Y_norm[i,idx] - Y_mean[i,0]
		self.Y = Y_norm
		self.Y_mean = Y_mean
	def callback(self,params):
		params_split = hsplit(params,(0,self.num_of_movies*self.num_of_features))
		self.X = reshape(params_split[1],(self.num_of_movies,self.num_of_features))
		self.Theta = reshape(params_split[2],(self.num_of_users,self.num_of_features))
	def predict_movie(self):
		self.normalizeY()
		(Param_opt,fopt,func_calls) = optimize.fmin_cg(
			f=lambda x: self.costFunction(x),
			x0=hstack([self.X.flatten(),self.Theta.flatten()]),
			fprime=lambda x: self.costFunctionGrad(x),
			disp=True,
			maxiter=self.MAX_ITERS,
			retall=True
		)
		param_opts = hsplit(Param_opt,(0,self.num_of_movies*self.num_of_features))
		X_opt = reshape(param_opts[1],(self.num_of_movies,self.num_of_features))
		Theta_opt = reshape(param_opts[2],(self.num_of_users,self.num_of_features))
		predict = X_opt.dot(transpose(Theta_opt))
		self.predict = predict + self.Y_mean
u = User()
print(u.predict[593,0])