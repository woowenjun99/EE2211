import matplotlib.pyplot as plt
import pandas as pd

class HomeworkOne():
    def __init__(self) -> None:
        df = pd.read_csv("government-expenditure-on-education.csv")
        self.x = df["year"].to_list()
        self.y = df["total_expenditure_on_education"].to_list()

    def plot_graph(self):
        plt.plot(self.x, self.y)
        plt.xlabel("year")
        plt.ylabel("total_expenditure_on_education")
        plt.title("Question 1")
        plt.show()

# Question 1
# h = HomeworkOne()
# h.plot_graph()


