import matplotlib.pyplot as plt
import pandas as pd

class HomeworkTwo():
    def __init__(self) -> None:
        pd.set_option('display.max_rows', 500)
        self.df = pd.read_csv("annual-motor-vehicle-population-by-vehicle-type.csv")
        self.years = self.df.loc[(self.df.type == "Omnibuses") & (self.df.year <= 2016)]["year"].to_list()
        pass

    def filter_data(self, type: str):
        return self.df.loc[(self.df.type == type) & (self.df.year <= 2016)]["number"].to_list()
    
    def plot(self):
        omnibuses = self.filter_data("Omnibuses")
        excursion = self.filter_data("Excursion buses")
        private = self.filter_data("Private buses")

        plt.plot(self.years, omnibuses, color="r", label="Omnibuses")
        plt.plot(self.years, excursion, color="b", label="Excursion")
        plt.plot(self.years, private, color="g", label="Excursion")
        plt.xlabel("year")
        plt.ylabel("Number of vehicles")
        plt.title("Number of vehicles over the years")
        plt.legend()
        plt.show()

h = HomeworkTwo()
h.plot()