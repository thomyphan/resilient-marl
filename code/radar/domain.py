import radar.environments.predator_prey as predator_prey

def make(domain, params={}):
    if domain.startswith("PredatorPrey-"):
        return predator_prey.make(domain, params)
    raise ValueError("Environment '{}' unknown or not published yet".format(domain))