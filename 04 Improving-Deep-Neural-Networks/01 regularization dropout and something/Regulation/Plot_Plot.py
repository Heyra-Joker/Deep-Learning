import matplotlib.pyplot as plt

def plot_plot(costs,acc_trains,acc_vals,val_costs,method):

    figure = plt.figure(figsize=(20,4))
    ax1 = figure.add_subplot(1,2,1)
    ax1.plot(costs,'-o',label="train loss")
    ax1.plot(val_costs,'-o',c='orange',label='val loss')
    ax1.set_title('{} Loss'.format(method))
    ax1.set_xlabel('#Iter')
    ax1.set_ylabel('Values')
    ax1.legend()
    
    ax2 = figure.add_subplot(1,2,2)
    ax2.plot(acc_trains,'-o',label="train accuracy")
    ax2.plot(acc_vals,'-o',c='orange',label="val accuracy")
    ax2.set_title('{} Accuracy'.format(method))
    ax2.set_xlabel('#Iter')
    ax2.set_ylabel('Values')
    ax2.legend()
    
    plt.show()
