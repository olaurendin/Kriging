from sklearn import metrics
from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score
from sklearn import cross_validation
import numpy as np
from numpy.random import uniform
from pykrige.core import _make_variogram_parameter_list, _initialize_variogram_model
from pykrige import variogram_models
from pykrige.rk import Krige
from rk_modified import Krige_modified
from pykrige.compat import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from decimal import Decimal

def random_relaxed_indexes(data, delta, nbtests, variogram_model,):

    variogram_dict = {'linear': variogram_models.linear_variogram_model,
                      'power': variogram_models.power_variogram_model,
                      'gaussian': variogram_models.gaussian_variogram_model,
                      'spherical': variogram_models.spherical_variogram_model,
                      'exponential': variogram_models.exponential_variogram_model,
                      'hole-effect': variogram_models.hole_effect_variogram_model}

    vp_temp = _make_variogram_parameter_list(variogram_model, None)
    lags, semivariances, vmp1 = \
    _initialize_variogram_model(data[:,:-1],data[:,-1], variogram_model=variogram_model,
            variogram_model_parameters=vp_temp,variogram_function=variogram_dict[variogram_model],nlags=15,weight=False, coordinates_type="euclidean")
    vmp11=[[vmp1[i]*(1-0.5*delta),vmp1[i]*(1+0.5*delta)]for i in range(len(vmp1))]
    print(vmp1, vmp11)
    l_parameters = np.array([[np.random.uniform(vmp11[i][0], vmp11[i][1]) for j in range(nbtests)] for i in range(len(vmp11))]).T

    return np.vstack((l_parameters,vmp1))


def CV(data, typesearch, delta, nbtests, nblags, cv):
    # print(data, delta, nbtests)

    dict_typesearch = {"GridSearchCV" : GridSearchCV,
                    "RandomizedSearchCV" : RandomizedSearchCV}

    variogram_model = "spherical"
    l_parameters = random_relaxed_indexes(data, delta, nbtests, variogram_model)
    print(l_parameters.tolist())
    l_parameters[:,0]+=l_parameters[:,2]

    param_dict = {"method": ["ordinary3d"], # , "universal3d"],
                  "variogram_model": ["spherical"],# ["linear", "power", "gaussian", "spherical"],
                  "nlags": [nblags],
                  # "variogram_parameters":[[61.47014547113474, 1197.0193021812333, 6.774575123767436]]
                  # "variogram_parameters":[[56.45916706474309, 1197.0193021812333, 9.138536040130816]]
                  # "weight": [True, False]
                  "variogram_parameters":l_parameters.tolist()
                  }
                  # best params : [63.350494982628305, 1150.332324073379, 6.4930511427290405]
    estimator = dict_typesearch[typesearch](Krige_modified(), \
        param_dict, verbose=True, cv=cv, return_train_score = True)

    X3 = data[:,:-1]
    y = data[:,-1]

    # run the gridsearch
    estimator.fit(X=X3, y=y)

    print(estimator.best_estimator_)
    print(estimator)


    if hasattr(estimator, 'best_score_'):
        print('best_score R² = {:.3f}'.format(estimator.best_score_))
        print('best_params = ', estimator.best_params_)

def test_gridsearch_cv_variogram_parameters():
    param_dict3d = {"method": ["ordinary3d"], "variogram_model": ["linear"],
                 "variogram_parameters": [{'slope': 1.0, 'nugget': 1.0},
                                          {'slope': 2.0, 'nugget': 1.0}]
               }

    estimator = GridSearchCV(Krige(), param_dict3d, verbose=True)

    # dummy data
    seed = np.random.RandomState(42)
    # X3 = seed.randint(0, 400, size=(100, 3)).astype(float)
    X3 = 400. * (1 + seed.rand(100, 3))
    y = 5 * seed.rand(100)

    # run the gridsearch
    estimator.fit(X=X3, y=y)

    # Expected best parameters
    best_params = [1.0,1.0]

    print("\n\n###########",estimator.cv_results_['param_variogram_parameters'], "\n\n", len(estimator.cv_results_['param_variogram_parameters']))

    if hasattr(estimator, 'best_score_'):
        print('best_score R² = {:.3f}'.format(estimator.best_score_))
        print('best_params = ', estimator.best_params_)
    # print("\n")
    # if hasattr(estimator2, 'best_score_'):
    #     print('best_score R² = {:.3f}'.format(estimator2.best_score_))
    #     print('best_params = ', estimator2.best_params_)
    #
    best_params = [1.0,1.0]
    # best_score = -0.4624793735893478
    best_score = round(Decimal(-0.462479373589),12)

    if hasattr(estimator, 'best_score_'):
        print(round(Decimal(estimator.best_score_),12), best_score)
        print("\n\n best score :", round(Decimal(estimator.best_score_),12) == best_score)
    if hasattr(estimator, 'best_params_'):
        print("\nlen param vario :", len(estimator.cv_results_['param_variogram_parameters'])==len(param_dict3d["variogram_parameters"]))
        for i,k in enumerate(estimator.best_params_["variogram_parameters"]):
            print("\n\negal parm vario :", estimator.best_params_["variogram_parameters"][k] == best_params[i])

if __name__=="__main__":
    test_gridsearch_cv_variogram_parameters()
