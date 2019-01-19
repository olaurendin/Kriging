def function_bidon(elm1, elm2, elm3):
    print(function_bidon.__dict__)
    print(function_bidon.__defaults__)
    print(elm1, elm2, elm3)

# def complete_function(f,params):
#     for k in params.keys():

# print(dir(function_bidon))
# print(function_bidon.__dict__)
# print(function_bidon.__defaults__)

# function_bidon.__dict__ = {"elm1":1, "elm2":2, "elm3":3}
# print(function_bidon.__dict__)
# print(function_bidon.__defaults__)

# function_bidon(4,5,6)
# function_bidon.__setattr__("elm1", 1)

d = {'elm1':1, 'elm2':2, 'elm3':3}

function_bidon(args = d)
