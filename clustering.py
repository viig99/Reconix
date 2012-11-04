from numpy import *
from scipy import linalg,optimize,io
import pdb,os

class User():
	def __init__(self):
		self.num_of_movies = 1682
		self.num_of_users = 943
		self.num_of_features = 4
		self.lambda_value = 5
		self.MAX_ITERS = 100
		self.Y = asmatrix(zeros((self.num_of_movies,self.num_of_users)))
		self.loadRatingsFileToArray('ml-100k/u.data')
		self.R = (self.Y != 0).astype(int)
		self.X = asmatrix(random.randn(self.num_of_movies,self.num_of_features))
		self.Theta = asmatrix(random.randn(self.num_of_users,self.num_of_features))
		self.predict_movie()
	def loadRatingsFileToArray(self,filename):
		f = open(filename)
		for line in f.readlines():
			ur = line.strip().split()
			self.Y[int(ur[1])-1,int(ur[0])-1] = ur[2]
		f.close()
	def costFunction(self,params):
		self.callback(params)
		J = (0.5 * nansum(multiply(power(self.X.dot(transpose(self.Theta)) - self.Y,2),self.R))) + ( (self.lambda_value/2)* (nansum(power(self.Theta,2)) + nansum(power(self.X,2))))
		print(J)
		return J
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
			Y_norm[i,idx] = self.Y[i,idx] - Y_mean[i,0]
		self.Y_mean = Y_mean
		self.Y = Y_norm
	def callback(self,params):
		params_split = hsplit(params,(0,self.num_of_movies*self.num_of_features))
		self.X = reshape(params_split[1],(self.num_of_movies,self.num_of_features))
		self.Theta = reshape(params_split[2],(self.num_of_users,self.num_of_features))
	def predict_movie(self):
		self.normalizeY()
		if not os.path.exists('predict.mat'):
			Param_opt = optimize.fmin_cg(
				f=lambda x: self.costFunction(x),
				x0=ravel(vstack((self.X,self.Theta))),
				fprime=lambda x: self.costFunctionGrad(x),
				disp=True,
				maxiter=self.MAX_ITERS,
				retall=True
			)
			param_opts = hsplit(Param_opt[0],(0,self.num_of_movies*self.num_of_features))
			X_opt = reshape(param_opts[1],(self.num_of_movies,self.num_of_features))
			Theta_opt = reshape(param_opts[2],(self.num_of_users,self.num_of_features))
			predict = X_opt.dot(transpose(Theta_opt))
			self.predict = predict + self.Y_mean
			io.savemat('predict.mat',{'predict':self.predict,'Y':self.Y})
		else:
			mat_contents = io.loadmat('predict.mat')
			self.predict = mat_contents['predict']
			self.R_not = logical_not(self.R).astype(int)
			self.predict_others = multiply(self.predict,self.R_not)
u = User()
print(u.predict[241,195])