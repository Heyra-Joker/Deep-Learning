from Softmax import  *
from  multiprocessing import Process,Queue,RLock
import os


def Save_plot(losses,val_losses,acc_trains,acc_vals,plot_path,lr):
    """
    Saving Plot to local.
    :param losses: train loss values, it's a list.
    :param val_losses: validation loss values, it's a list.
    :param acc_trains: train accuracy, list.
    :param acc_vals: validation data set values, list.
    :param plot_path: save plot image path.
    :param lr: learning rate
    :return: None
    """
    figure = plt.figure(figsize=(15, 4))
    ax1 = figure.add_subplot(1, 2, 1)
    ax1.plot(losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_title('{}:Loss'.format(lr))
    ax1.set_xlabel('#Iterate')
    ax1.set_ylabel('Values')
    ax1.legend()

    ax2 = figure.add_subplot(1, 2, 2)
    ax2.plot(acc_trains, label="Train Acc")
    ax2.plot(acc_vals, label="Val Acc")
    ax2.set_title('{}:Accuracy'.format(lr))
    ax2.set_xlabel('#Iterate')
    ax2.set_ylabel('Values')
    ax2.legend()
    plt.savefig(plot_path)


def Master(*args):
    """
    Note: 我们这里的随机采取最好是偶数个,因为多进程如果是奇数个的话效率会很慢.
    """
    for _ in range(4):
        r = np.random.uniform(search_start,search_end)
        q.put(r)



def Workers(*args):
    """
    Processing of Works.Complete running model at Master given learning rate.

    """

    while 1:
        print('Process:{} Start!'.format(os.getpid()))
        # get lr.
        lr = np.round(np.power(10,q.get()),4)
        print('Process:{} trying lr:{}'.format(os.getpid(),lr))
        losses, val_losses, acc_trains, acc_vals, parameters,SON_PATH = Softmax_Model(layers, x_train, y_train, x_test, y_test,
                                                                             lr=lr,
                                                                             epochs=20,
                                                                             beta_1=0.9,
                                                                             beta_2=0.999,
                                                                             batc_size=64,
                                                                             save_path=save_path,
                                                                             lock=lock)
        plot_path = SON_PATH + str(lr) + '.jpg'
        Save_plot(losses,val_losses,acc_trains,acc_vals,plot_path,lr)

        if q.empty():
            print('Process:{} Over!'.format(os.getpid()))
            break


if __name__ == "__main__":
    # get search range,In this case, we use -4 to -2.
    search_start,search_end = eval(input('Lr Search Range:>>'))
    save_path = 'log/'
    ########################### Load Dataset ###################
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1) / 255
    x_test = x_test.reshape(x_test.shape[0], -1) / 255
    ########################### end ############################

    layers = [25, 12, 10]

    print("Searching Parameters...")
    lock = RLock()
    q = Queue()
    p1 = Process(target=Master,args=(q,search_start,search_end))
    p2 = Process(target=Workers,args=(q,layers, x_train, y_train, x_test, y_test,save_path,lock))
    p3 = Process(target=Workers,args=(q,layers, x_train, y_train, x_test, y_test,save_path,lock))

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()


    print("Searching OK")

