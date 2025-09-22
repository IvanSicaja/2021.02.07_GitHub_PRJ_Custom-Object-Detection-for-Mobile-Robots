import datetime       # Module used for print current date and time

currentTimeAndDate=datetime.datetime.now()
print("Current date and time is: ")
print(currentTimeAndDate.strftime("%y-%m-%d_%H:%M:%S") )

currentTimeAndDate=currentTimeAndDate.strftime("%y-%m-%d_%H:%M:%S")
print(currentTimeAndDate)