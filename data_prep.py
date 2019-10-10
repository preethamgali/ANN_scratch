# day data prepare

# temp    hum     windspeed      clear_d  cloudy   rainy  h_rain    working_day      holiday
# out     casual_riders     registered_riders
import numpy as np


data = []
with open("day.csv") as f:
    lines = f.readlines()

    for line in lines[1:]:
        line = line.split(',')
        data_row =[]
        working_day = [int(line[7])]
        weather = [0]*4
        weather[int(line[8])-1] = 1
        data_row = working_day + weather + [float(x) for x in line[10:-3]]
        data.append(data_row)

    c_users = np.asarray([x[-2] for x in data])
    r_users = np.asarray([x[-1] for x in data])
    c_mean, c_std, c_max = np.mean(c_users), np.std(c_users), np.max(c_users)
    r_mean, r_std, r_max = np.mean(r_users), np.std(r_users), np.max(r_users)

    c_users = (c_users-c_mean)/c_max
    r_users = (r_users-r_mean)/r_max

    output = []
    for i,j in zip(c_users,r_users):
        output.append([i,j]) 

    # for d in data:
    #     d[-2]=(c_mean-d[-2])/c_std
    #     d[-1]=(r_mean-d[-1])/r_std
 
# "working_day,clear_d,cloudy_d,rainy_d,heavy_d,atem,hum,wind_s,casual,registered\n"

# with open("data.csv",'w+') as data_file:
#     data_file.write("working_day,clear_d,cloudy_d,rainy_d,heavy_d,atem,hum,wind_s,casual,registered\n")
#     for x in data:
#         line = ''
#         for i in x:
#             line+= str(i)+','
#         line=line[:-1]+'\n'
#         data_file.write(line)






    

