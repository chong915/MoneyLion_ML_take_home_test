from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score


def hyperparamter_tuning(X_train, X_val, y_train, y_val):
    # Create LightGBM datasets
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    # Hyperparameter tuning with hyperopt and lightgbm
    def objective(params):
        params['objective'] = 'binary'
        params['metric'] = 'auc'
        params['verbosity'] = -1

        # Remove n_estimators from params, because we pass it as a function argument
        n_estimators = params['n_estimators']
        del params['n_estimators']
        params['num_leaves'] = int(params['num_leaves'])
        
        model = lgb.train(params, train_set=dtrain, valid_sets=[dval], num_boost_round=n_estimators,callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)])
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        preds_binary = np.round(preds)  # Convert probabilities to binary
        f1 = f1_score(y_val, preds_binary)
        return {'loss': -f1, 'status': STATUS_OK}

    space = {
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
        'lambda_l2': hp.quniform('lambda_l2', 0, 10, 1),
        'num_leaves': hp.quniform('num_leaves', 30, 80, 5),
        'n_estimators': hp.choice('n_estimators', range(50, 500)),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
    }

    trials = Trials()
    best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials, rstate=np.random.default_rng(42))

    print("Best Parameters:", best_params)

    return best_params