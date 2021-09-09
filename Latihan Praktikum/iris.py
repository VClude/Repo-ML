# part 1

import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('iris.csv')
print (data.head(10)) 
data.describe()


#part 2

plt.figure(figsize = (10, 7)) 
x = data["sepal.length"] 
  
plt.hist(x, bins = 20, color = "green") 
plt.title("Sepal Length in cm") 
plt.xlabel("Sepal_Length_cm") 
plt.ylabel("Count") 

plt.show()

#part 3

# show the box plot
new_data = data[["sepal.length", "sepal.width", "petal.length", "petal.width"]] 
print(new_data.head()) 

plt.figure(figsize = (10, 7)) 
new_data.boxplot() 

plt.show() 
