import random
import numpy as np

# Function to calculate the fitness (closeness to pi) of an individual
def calculate_fitness(individual):
    x = individual[0]
    y = individual[1]
    distance = np.sqrt(x**2 + y**2)
    return abs(distance - 1.0)

# Function to generate initial population
def generate_population(size):
    return [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(size)]

# Function to perform crossover between two individuals
def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

# Function to perform mutation on an individual
def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.uniform(-1, 1)
    return individual

# Genetic algorithm to approximate pi
def approximate_pi(population_size, generations, mutation_rate):
    population = generate_population(population_size)
    best_approximation = [float('inf'), None]  # [Best Fitness, Individual] -> best fitness assigned as infinity at first

    for generation in range(generations):
        fitness_scores = [calculate_fitness(individual) for individual in population]

        # Find the individual with the lowest fitness (closest to pi)
        best_index = fitness_scores.index(min(fitness_scores))
        best_individual = population[best_index]
        best_fitness = fitness_scores[best_index]

        # Update best approximation if a better one is found
        if best_fitness < best_approximation[0]:
            best_approximation = [best_fitness, best_individual, generation]

        # Output the best approximation in each generation
        print(f"Generation {generation}: Best approximation to pi = {np.pi - best_fitness}")

        # Create the next generation using crossover and mutation
        new_population = []
        for _ in range(population_size):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

    return best_approximation

# Parameters
population_size = 1000
generations = 10000
mutation_rate = 0.1

# Run the genetic algorithm
best_pi_info = approximate_pi(population_size, generations, mutation_rate)
best_pi_fitness, best_pi_individual, best_pi_generation = best_pi_info
print(f"Best approximation of pi ({np.pi - best_pi_fitness}) found at generation {best_pi_generation}")
