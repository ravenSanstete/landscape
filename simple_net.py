import autograd.numpy as np
from autograd import grad
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

## define a simple neural net, of dimension 2
def relu_nn(w):
    relu = lambda y: np.float32(y >= 0) * y;
    W_mat = np.array([[w[0], w[1]], [w[1], w[0]]]);
    return (lambda x: np.dot(relu(np.matmul(W_mat, x)), w));

def ind(w, bd, delta, res):
    candi = np.int32((w -bd) / delta);
    if(np.any(candi < 0) or np.any(candi >= res)):
        return np.array([-1, -1], dtype= np.int32);
    else:
        return candi;
    

    
    
if __name__ == '__main__':
    w_star = np.random.randn(2);
    t_net = relu_nn(w_star); # fix the teacher net
    
    alpha = 0.005; ## step size
    gamma = 0.01; ## perturbance
    w = 2.0*np.random.randn(2)

    # the scan parameter, it means we divide the [-bd, bd]^2 to res*res blocks
    res = 100
    dist = np.abs(w - w_star);
    bd = w_star - dist;
    bd_hi = w_star + dist;
    delta = 2*dist / res;

    n_grid = np.zeros([res, res], dtype = np.int32);
    l_grid = np.zeros([res, res]);
    print("Target ({},{})".format(w_star[0], w_star[1]));
    max_iter = 10000;
    ## the student network begins to scan the neural network
    for i in range(max_iter):
        x = np.random.randn(2);
        s_net = relu_nn(w)
        loss = lambda x: (s_net(x) - t_net(x))**2;
        w = w - alpha * (grad(loss))(x) + gamma*np.random.randn(2);
        j = ind(w, bd, delta, res);
        if(j[0] >= 0):
            n_t = n_grid[j[0]][j[1]];
            l_grid[j[0]][j[1]] = (n_t*l_grid[j[0]][j[1]] + loss(x)) / (n_t + 1);
            n_grid[j[0]][j[1]] = n_t+1;
        else:
            w = 2.0*np.random.randn(2)
        if(np.mod(i, 100) == 0):
            print("Iter {} Weight ({},{}) Loss {} Ind ({}, {})".format(i, w[0], w[1], loss(x), j[0], j[1]));

    print("scan ratio {}".format(np.mean(n_grid)));
    # x_0 = np.random.randn(2);
    # print(t_net(x_0))
    # print((grad(t_net)(x_0)))
    x_line = np.arange(bd[0], bd_hi[0], delta[0])[0:res];
    y_line = np.arange(bd[1], bd_hi[1], delta[1])[0:res];
    x_grid, y_grid = np.meshgrid(x_line, y_line);
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(x_grid, y_grid, l_grid);
    plt.show();
    