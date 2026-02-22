from deap import base, creator, tools, algorithms
import random
import numpy as np
from functools import partial
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score

def get_sigma_value(series):
    """
    Gets the value of the data point just after the 10-sigma threshold.
    If the time series does not exceed the 10-sigma threshold, returns 0.

    Args:
        series (pd.Series): The time series data.

    Returns:
        float: The value of the data point just after the 10-sigma threshold.
    """
    first_10 = series[:10]
    mean = np.mean(first_10)
    std = np.std(first_10)
    sigma_time = mean + 10 * std
    index = np.argmax(series > sigma_time)
    if index == 0:
      value = 0
    else:
      value = series[index]
    return value

def get_sigma_index(series, num_base_count=10, sigma_multiplier=10):
    """
    Gets the index of the data point just after the 10-sigma threshold.
    If the time series does not exceed the 10-sigma threshold, returns 0.

    Args:
        series (pd.Series): The time series data.

    Returns:
        float: The index of the data point just after the 10-sigma threshold.
    """
    bases = series[:num_base_count]
    mean = np.mean(bases)
    std = np.std(bases)
    sigma_time = mean + sigma_multiplier * std
    index = np.argmax(series > sigma_time)
    return index

def thresholding(X, threshold, y, num_base_count=10, sigma_multiplier=10, index_time_scale=1):
    """
    Performs thresholding with a specific threshold. But this time, it returns the sensitivity, specificity and ROC-AUC.

    Args:
        X (pd.DataFrame): The time series data.
        threshold (float): The threshold value.
        outcomes (pd.Series): The outcomes of the time series.
        index_time_scale (int, optional): The time scale of the index. Defaults to 1.

    Returns:
        pd.DataFrame: A new DataFrame with the sigma value, sigma time, and whether the threshold was met.
    """
    X['sigma_time'] = X['timeseries'].apply(
        lambda x: get_sigma_index(x, num_base_count=num_base_count,
                                  sigma_multiplier=sigma_multiplier)) / index_time_scale
    X['meets_threshold'] = (X['sigma_time'] <= threshold).astype(int) # 1 for positive, 0 for negative

    y_pred = X['meets_threshold']

    cm = confusion_matrix(y, y_pred)

    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    auc = roc_auc_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)

    #youden_j = sensitivity + specificity - 1
    balance = 1 - abs(sensitivity - specificity)

    return sensitivity, specificity, balance, auc

def evaluate(individual, X_train, y_train, index_time_scale):
  num_base_count = individual[0]
  sigma_multiplier = individual[1]
  threshold = individual[2]

  return thresholding(X_train.copy(), threshold, y_train.copy(), num_base_count=num_base_count, sigma_multiplier=sigma_multiplier, index_time_scale=index_time_scale)

def cxMixed(ind1, ind2, eta=20):
    # --- continuous parameters ---
    # wrap continuous parameters in lists for cxSimulatedBinary
    continuous_params1 = [ind1[1]]
    continuous_params2 = [ind2[1]]
    tools.cxSimulatedBinary(continuous_params1, continuous_params2, eta=eta)
    ind1[1] = continuous_params1[0]
    ind2[1] = continuous_params2[0]


    # --- categorical parameters ---
    # num base count
    if random.random() < 0.5:
        ind1[0], ind2[0] = ind2[0], ind1[0]

    # threshold
    if random.random() < 0.5:
        ind1[2], ind2[2] = ind2[2], ind1[2]

    return ind1, ind2

def get_indpb(gen, max_gen, start=0.4, end=0.1):
    """Linearly decay mutation probability from start to end"""
    return start - (start - end) * (gen / max_gen)

def mutate_continuous(individual, mu=0, sigma=0.1, indpb=0.3):
    """Gaussian mutation for continuous params"""
    if random.random() < indpb:
        individual[1] += random.gauss(mu, sigma)
        individual[1] = max(individual[1], 0)  # ensure no negative values
    return individual

def mutate_categorical(individual, indpb=0.3):
    """Uniform mutation for categorical params"""
    # num base count
    if random.random() < indpb:
        individual[0] = random.randint(3, 15)
    # threshold
    if random.random() < indpb:
        individual[2] = random.randint(3, 30)

    return individual

def mutate(individual, indpb):
    individual = mutate_continuous(individual, indpb=indpb)
    individual = mutate_categorical(individual, indpb=indpb)
    return individual,

def perform_thresholding(X_train, y_train, index_time_scale, pop_size=200, max_gen=100):
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, 1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # attribute generators
    toolbox.register("attr_num_base_count", random.randint, 3, 15)
    toolbox.register("attr_sigma_multiplier", random.uniform, 5.0, 15.0)
    toolbox.register("attr_threshold", random.randint, 3, 30)
    
    # structure initialisers
    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_num_base_count, toolbox.attr_sigma_multiplier, toolbox.attr_threshold), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # genetic algorithm operators
    toolbox.register("evaluate", partial(evaluate, X_train=X_train, y_train=y_train, index_time_scale=index_time_scale))
    toolbox.register("mate", cxMixed, eta=20)
    toolbox.register("mutate", mutate, indpb=0.3)
    toolbox.register("select", tools.selNSGA2)
    
    pop = toolbox.population(n=pop_size)
    hof = tools.ParetoFront()  # hall of fame

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)


    result, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size, cxpb=0.7, mutpb=0.3,
                                         ngen=max_gen, stats=stats, halloffame=hof,
                                         verbose=True)

    return pop, hof, logbook

def get_thresholding_values(X, threshold, outcomes, num_base_count=10, sigma_multiplier=10, index_time_scale=1):
    X['sigma_time'] = X['timeseries'].apply(
        lambda x: get_sigma_index(x, num_base_count=num_base_count,
                                  sigma_multiplier=sigma_multiplier)) / index_time_scale
    X['meets_threshold'] = (X['sigma_time'] <= threshold).astype(int)

    return X

    return pop, hof, logbook