import numpy as np
import matplotlib.pyplot as plt

# 读取txt文件并将其转换为numpy矩阵
train_data = np.loadtxt('./runs/train.txt')
test_data = np.loadtxt('./runs/test.txt')

# 打印读取的矩阵
print(train_data.shape) # choose loss 5 and dice 1, need a

dice = np.zeros((train_data.shape[0],2))
dice[:,0] = train_data[:,1]
dice[:,1] = test_data[:,1]

loss = np.zeros((train_data.shape[0],2))
loss[:,0] = train_data[:,5]
loss[:,1] = test_data[:,5]

x_axis = np.arange(1,train_data.shape[0]+1)
plt.figure()
plt.plot(x_axis, dice[:,0],"b-",label="train")
plt.plot(x_axis, dice[:,1],"r-",label="test")
plt.grid()
plt.legend()
plt.xlabel("epoch")
plt.ylabel("dice value")
plt.title("Dice | simple U-net + t1 channel")
plt.savefig("dice.png")
plt.show()
plt.pause(3)
plt.close()


plt.figure()
plt.plot(x_axis, loss[:,0],"b-",label="train")
plt.plot(x_axis, loss[:,1],"r-",label="test")
plt.grid()
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss value")
plt.title("Binary Cross Entropy Loss | simple U-net + t1 channel")
plt.savefig("loss.png")
plt.show()
plt.pause(3)
plt.close()