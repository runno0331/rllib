import matplotlib.pyplot as plt

class Logger:
    def __init__(self):
        self.loss_record = []
        self.total_reward_record = []

    def update(self, loss=None, total_reward=None):
        if loss is not None:
            self.loss_record.append(loss)
        if total_reward is not None:
            self.total_reward_record.append(total_reward)

    def show_loss_record(self):
        plt.plot(self.loss_record)
        plt.ylabel('Loss')
        plt.show()

    def show_total_reward_record(self):
        plt.plot(self.total_reward_record)
        plt.ylabel('Total reward')
        plt.show()