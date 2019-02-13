# Understanding Independent Component Analysis (ICA)
# The code is from https://blog.paperspace.com/dimension-reduction-with-
## independent-components-analysis/

import numpy as np

x = np.array([-5,-4,-2,-1,0,1,2,3,4,5])
y = np.array([25,16,9,4,1,0,1,4,9,16,25])

np.correlate(x,y)

np.random.seed(100)
U1 = np.random.uniform(-1, 1, 1000)
U2 = np.random.uniform(-1, 1, 1000)

G1 = np.random.randn(1000)
G2 = np.random.randn(1000)

%matplotlib inline
# let's plot our signals

from matplotlib import pyplot as plt

fig = plt.figure()

ax1 = fig.add_subplot(121, aspect = "equal")
ax1.scatter(U1, U2, marker = ".")
ax1.set_title("Uniform")


ax2 = fig.add_subplot(122, aspect = "equal")
ax2.scatter(G1, G2, marker = ".")
ax2.set_title("Gaussian")


plt.show()


# now comes the mixing part. we can choose a random matrix for the mixing

A = np.array([[1, 0], [1, 2]])

U_source = np.array([U1,U2])
U_mix = U_source.T.dot(A)

G_source = np.array([G1, G2])
G_mix = G_source.T.dot(A)

# plot of our dataset

fig  = plt.figure()

ax1 = fig.add_subplot(121)
ax1.set_title("Mixed Uniform ")
ax1.scatter(U_mix[:, 0], U_mix[:,1], marker = ".")

ax2 = fig.add_subplot(122)
ax2.set_title("Mixed Gaussian ")
ax2.scatter(G_mix[:, 0], G_mix[:, 1], marker = ".")


plt.show() 

# PCA and whitening the dataset
from sklearn.decomposition import PCA 
U_pca = PCA(whiten=True).fit_transform(U_mix)
G_pca = PCA(whiten=True).fit_transform(G_mix)

# let's plot the uncorrelated columns from the datasets
fig  = plt.figure()

ax1 = fig.add_subplot(121)
ax1.set_title("PCA Uniform ")
ax1.scatter(U_pca[:, 0], U_pca[:,1], marker = ".")

ax2 = fig.add_subplot(122)
ax2.set_title("PCA Gaussian ")
ax2.scatter(G_pca[:, 0], G_pca[:, 1], marker = ".")

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
np.random.seed(0)
num_rows = 3000
t = np.linspace(0,10, num_rows)
# create signals sources
s1 = np.sin(3*t) # a sine wave
s2 = np.sign(np.cos(6*t)) # a square wave
s3 = signal.sawtooth(2 *t) # a sawtooth wave
# combine single sources to create a numpy matrix
S = np.c_[s1,s2,s3]

# add a bit of random noise to each value
S += 0.2* np.random.normal(size = S.shape)

# create a mixing matrix A
A = np.array([[1, 1.5, 0.5], [2.5, 1.0, 2.0], [1.0, 0.5, 4.0]])
X = S.dot(A.T)

#plot the single sources and mixed signals
plt.figure(figsize =(26,12) )
colors = ['red', 'blue', 'orange']

plt.subplot(2,1,1)
plt.title('True Sources')
for color, series in zip(colors, S.T):
    plt.plot(series, color)
plt.subplot(2,1,2)
plt.title('Observations(mixed signal)')
for color, series in zip(colors, X.T):
    plt.plot(series, color)

## Notice the differences between the uncorrelated(PCA uniform, PCA gaussian2)
## and source plots(Uniform, Gaussian). In case of Gaussian they look alike while 
## uncorrelated Uniform needs a rotation to get there. By removing correlation
## in the gaussian case, we have achieved independence between variables.
## If the source variables are gaussian ICA is not required and PCA is sufficient.
    
    
# Code for PCA and whitening the dataset.

from pyspark.mllib.linalg.distributed import IndexedRowMatrix, IndexedRow, BlockMatrix
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.linalg import Vectors, DenseMatrix, Matrix
from sklearn import datasets
# create the standardizer model for standardizing the dataset

X_rdd = sc.parallelize(X).map(lambda x:Vectors.dense(x) )
scaler = StandardScaler(withMean = True, withStd = False).fit(iris_rdd)

X_sc = scaler.transform(X_rdd)


#create the IndexedRowMatrix from rdd
X_rm = IndexedRowMatrix(X_sc.zipWithIndex().map(lambda x: (x[1], x[0])))

# compute the svd factorization of the matrix. First the number of columns and second a boolean stating whether 
# to compute U or not. 
svd_o = X_rm.computeSVD(X_rm.numCols(), True)

# svd_o.V is of shape n * k not k * n(as in sklearn)

P_comps = svd_o.V.toArray().copy()
num_rows = X_rm.numRows()
# U is whitened and projected onto principal components subspace.

S = svd_o.s.toArray()
eig_vals = S**2
# change the ncomp to 3 for this tutorial
#n_comp  = np.argmax(np.cumsum(eig_vals)/eig_vals.sum() > 0.95)+1
n_comp = 3
U = svd_o.U.rows.map(lambda x:(x.index, (np.sqrt(num_rows-1)*x.vector).tolist()[0:n_comp]))
# K is our transformation matrix to obtain projection on PC's subspace
K = (U/S).T[:n_comp]

import pyspark.sql.functions as f
import pyspark.sql.types as t

df = spark.createDataFrame(U).toDF("id", "features")


# Approximating function g(y) = x*exp(-x**2/2) and its derivative
def g(X):
    x = np.array(X)
    return(x * np.exp(-x**2/2.0))

def gprime(Y):
    y = np.array(Y) 
    return((1-y**2)*np.exp(-y**2/2.0))

# function for calculating step 2 of the ICA algorithm
def calc(df):
    
    ## function to calculate the appoximating function and its derivative
    def foo(x,y):

        y_arr = np.array(y)
        gy = g(y_arr)
        gp = gprime(y_arr)
        x_arr = np.array(x)
        res = np.outer(gy,x_arr)
        return([res.flatten().tolist(), gp.tolist()])

    udf_foo = f.udf(foo, t.ArrayType(t.ArrayType(t.DoubleType())))



    df2 = df.withColumn("vals", udf_foo("features","Y"))

    df2 = df2.select("id", f.col("vals").getItem(0).alias("gy"), f.col("vals").getItem(1).alias("gy_"))
    GY_ = np.array(df2.agg(f.array([f.sum(f.col("gy")[i]) 
                                for i in range(n_comp**2)])).collect()[0][0]).reshape(n_comp,n_comp)/num_rows

    GY_AVG_V  = np.array(df2.agg(f.array([f.avg(f.col("gy_")[i]) 
                                  for i in range(n_comp)])).collect()[0][0]).reshape(n_comp,1)*V

    return(GY_, GY_AVG_V)


np.random.seed(101)
# Initialization
V = np.random.rand(n_comp, n_comp)

# symmetric decorelation function 
def sym_decorrelation(V):

    U,D,VT = np.linalg.svd(V)
    Y = np.dot(np.dot(U,np.diag(1.0/D)),U.T)
    return np.dot(Y,V)

numIters = 10
V = sym_decorrelation(v_init)
tol =1e-3
V_bc = sc.broadcast(V)

for i in range(numIters):

    # Y = V*X
    udf_mult = f.udf(lambda x: V_bc.value.dot(np.array(x)).tolist(), t.ArrayType(t.DoubleType()))
    df = df.withColumn("Y", udf_mult("features"))

    gy_x_mean, g_y_mean_V = calc(df)

    V_new = gy_x_mean - g_y_mean_V

    V_new = sym_decorrelation( V_new )

    #condition for convergence
    lim = max(abs(abs(np.diag(V_new.dot(V.T)))-1))

    V = V_new
    # V needs to be broadcasted after every change
    V_bc = sc.broadcast(V)

    print("i= ",i," lim = ",lim)

    if lim < tol:
        break
    elif i== numIters:
        print("Lower the tolerance or increase the number of iterations")

#calculate the unmixing matrix for dataset         
W = V.dot(K)

#now multiply U with V to get source signals
S_ = df.withColumn("Y", udf_mult("features"))

plt.title('Recovered source Signals')
for color, series in zip(colors, S_.T):
    plt.plot(series, color)

