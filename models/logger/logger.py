import matplotlib.pyplot as plt

'''
    Takes log of loss and total reward
'''


class Logger:
    def __init__(self):
        self.loss_record = []
        self.total_reward_record = []

    # push record data
    def update(self, loss=None, total_reward=None):
        if loss is not None:
            self.loss_record.append(loss)
        if total_reward is not None:
            self.total_reward_record.append(total_reward)

    # show loss graph
    def show_loss_record(self):
        plt.plot(self.loss_record)
        plt.ylabel('Loss')
        plt.show()

    # show reward graph
    def show_total_reward_record(self):
        plt.plot(self.total_reward_record)
        plt.ylabel('Total reward')
        plt.show()