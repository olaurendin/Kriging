def fonction_a_completer(param1, param2, param3):
    return param1, param2, param3

dict_params = {"param1":1,"param2":2,"param3":3}
dict_params2 = {"param2":2,"param3":3}

def complete_function(f, args=False, kwargs=False):
    # if type(args) is not list and args is not False :
    #     args = [args]
    if args is not False and kwargs is False :
        print("args")
        return f(*args)
    elif args is False and kwargs is not False :
        print("kwargs")
        return f(**kwargs)
    elif args is not False and kwargs is not False :
        print("argskwargs")
        return f(*args, **kwargs)
    else:
        raise ValueError("Parameters for the function '{}' incomplete. Please \
check the input list '*args' and/or dictionary parameters '**kwargs' of the function."
        .format(f.__name__))

print(complete_function(fonction_a_completer,  args = [1,2,3])) #, kwargs = dict_params)) #  args = [1,2,3], kwargs = dict_params))
print(complete_function(fonction_a_completer,  args = [1], kwargs = dict_params2))
print(complete_function(fonction_a_completer,  kwargs = dict_params))
print(complete_function(fonction_a_completer))
