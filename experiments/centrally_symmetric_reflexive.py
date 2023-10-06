import random
import datetime
import os
import numpy as np
from sympol import Polytope
from sympol.ehrhart import h_star_vector_of_cartesian_product_from_h_star_vectors


def _generate_all_mutations_single(p: Polytope):
    """Generate all possible mutations of h."""

    mutations = set()

    # take the product with [-1,1]
    q = p * (Polytope.cross_polytope(1))
    q._is_reflexive = True
    q._h_star_vector = h_star_vector_of_cartesian_product_from_h_star_vectors(
        p._h_star_vector, tuple([1, 1])
    )
    mutations.add(q)

    p_pts = p.integer_points
    # remove couple of opposite vertices
    for v in p.vertices:
        # avoid removing the same vertices twice
        nzv = [a for a in v if a != 0]
        if nzv[0] < 0:
            continue

        # remove the vertex and its opposite
        v_plus = tuple([a for a in v])
        v_minus = tuple([-a for a in v])
        new_verts = [u for u in p_pts if tuple(u) not in [v_plus, v_minus]]
        q = Polytope(new_verts)
        mutations.add(q)

        new_verts = [u for u in p.vertices if tuple(u) not in [v_plus, v_minus]]
        q = Polytope(new_verts)
        mutations.add(q)

        # ... or move a couple of opposite vertices
        # get a vecttor with random 1-,0,1 entries
        u = np.array([random.randint(-1, 1) for _ in range(len(v))])
        v = np.array(v)
        new_verts = new_verts + [tuple(v + u), tuple(-v - u)]

        q = Polytope(new_verts)
        # if q.is_reflexive:
        mutations.add(q)

    return mutations


def _generate_all_crossovers_single(p1, p2):
    pass
    # cs = set()

    # # in case the lengths do not match, add both and return
    # if len(h1) <= len(h2):
    #     g1 = h1
    #     g2 = h2
    # elif len(h1) > len(h2):
    #     g1 = h2
    #     g2 = h1

    # while len(g1) < len(g2):
    #     g1 = g1 + (0,)

    # # in case the lengths match generate all intermediate vectors
    # ranges = [range(min(g1[i], g2[i]), max(g1[i], g2[i]) + 1) for i in range(len(g1))]

    # # generate all possible combinations of the diffs
    # for h in itertools.product(*ranges):
    #     cs.add(h)

    # return cs


def _remove_unfit_single(mutants):
    to_remove = set()
    for p in mutants:
        if not p.is_reflexive:
            to_remove.add(p)

    for p in to_remove:
        mutants.remove(p)

    return mutants


def _calculate_fitness_function_single(p: Polytope):
    # lower is better
    score = 1

    gamma = p.gamma_vector

    # last nonzero entry
    last = max([i for i, g_i in enumerate(gamma) if g_i != 0])
    gamma = gamma[: last + 1]

    for _, g_i in enumerate(gamma):
        penalty = max(0, g_i + 1)
        score *= penalty

    score = score ** (1 / len(gamma))

    return score


def genetic_search(
    parameters,
):
    """ """
    # read parameters
    population = parameters["starting_population"]
    generate_all_mutations = parameters["generate_all_mutations"]
    generate_all_crossovers = parameters["generate_all_crossovers"]
    remove_unfit = parameters["remove_unfit"]
    fitness_function = parameters["fitness_function"]
    p_mutation = parameters["p_mutation"]
    max_pop = parameters["max_pop"]
    min_pop = parameters["min_pop"]
    penalty_factor = parameters["penalty_factor"]

    # create logging folder
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = ".logs"
    os.makedirs(folder_name, exist_ok=True)
    filename = f"{folder_name}/{timestamp}_search.txt"

    # init the score dictionary, will be updated at each epoch with penalties
    scores = {}
    for h in population:
        scores[h.gamma_vector] = fitness_function(h)

    generation = 0

    # log the first values
    with open(filename, "a") as f:
        f.write(
            f"{generation}\t{h._gamma_vector}\t{scores[h._gamma_vector]}\t{scores[h._gamma_vector]}\n"
        )

    while True:
        # start a new generation
        generation += 1

        # generate new mutations
        n_mutation_attempts = 0
        while len(population) < max_pop:
            n_mutation_attempts += 1

            # pick between mutation and crossover depending on probabilities
            if random.random() < p_mutation:
                # attempt mutation

                # pick an h randomly, but weighted on the (reciprocal of) fitness score
                total_score = sum([1 / scores[h._gamma_vector] for h in population])
                probs = [
                    1 / (scores[h._gamma_vector] * total_score) for h in population
                ]
                h = random.choices(population, weights=probs, k=1)[0]

                # generate all possible mutations for h
                possible_mutations = set(generate_all_mutations(h))

                # remove all already existing mutations and unfit mutations
                possible_mutations = possible_mutations.difference(set(population))
                possible_mutations = remove_unfit(possible_mutations)

                # if there are no possible mutations, penalize h and continue
                if len(possible_mutations) == 0:
                    # scores[h] *= penalty_factor
                    continue

                # pick a mutation randomly and add it to the population
                p_new = random.choice(tuple(possible_mutations))
            else:
                # attempt crossover

                # pick two h's randomly, but weighted on the (reciprocal of) fitness score
                total_score = sum([1 / scores[h._gamma_vector] for h in population])
                probs = [
                    1 / (scores[h._gamma_vector] * total_score) for h in population
                ]
                h1, h2 = random.choices(population, weights=probs, k=2)

                # generate all possible crossovers for h1 and h2
                possible_crossovers = generate_all_crossovers(h1, h2)

                # remove all already existing crossovers and unfit crossovers
                possible_crossovers = possible_crossovers.difference(set(population))
                possible_crossovers = remove_unfit(possible_crossovers)

                # if there are no possible crossovers, penalize h1 and h2 and continue
                # (only do this if len(h1) == len(h2))
                if len(possible_crossovers) == 0:
                    # if len(h1) == len(h2):
                    # scores[h1] *= penalty_factor
                    # scores[h2] *= penalty_factor
                    continue

                # pick a crossover randomly and add it to the population
                p_new = random.choice(tuple(possible_crossovers))

            if not h.is_reflexive:
                continue

            # calculate the gamma vector and th score
            score_p_new = fitness_function(p_new)

            # check if h is a solution, if so, write to the logs and return it
            if score_p_new == 0:
                print(f"Found eligible mutation: {p_new.vertices}")
                avg_score = sum(sorted(scores.values())[:min_pop]) / min_pop
                with open(filename, "a") as f:
                    f.write(f"{generation}\t{p_new}\t{score_p_new}\t{avg_score}\n")
                return p_new

            # add p_new to the population
            population.append(p_new)
            if p_new._gamma_vector not in scores:
                scores[p_new._gamma_vector] = score_p_new

        # now start the selection process, pick the best min_pop individuals based on
        # their fitness function score (lower is better)
        selected = sorted(population, key=lambda h: scores[h._gamma_vector])[:min_pop]

        # find the best of the elected for logging purposes
        best_h_score = min([scores[h._gamma_vector] for h in selected])
        best_h = [h for h in selected if scores[h._gamma_vector] == best_h_score][0]

        # find the average score
        avg_score = sum([scores[h._gamma_vector] for h in selected]) / len(selected)

        # log generation summary to console
        print(f"Generation {generation} summary:")
        print(f"Mutation attempt: {n_mutation_attempts}")
        print(f"Average score: {avg_score}")
        print(f"Best score: {best_h_score}")
        print(f"Best h: {best_h._h_star_vector}")
        print(f"Best gamma: {best_h._gamma_vector}")

        print()

        # log generation summary to file
        with open(filename, "a") as f:
            f.write(f"{generation}\t{best_h}\t{best_h_score}\t{avg_score}\n")

        # check if h is a solution, if so, write to the logs and return it
        if best_h_score == 0:
            print(f"Found eligible mutation: {p_new}")
            return best_h

        # multiply all scores by the penality factor (aging process)
        for h in population:
            scores[h._gamma_vector] *= penalty_factor

        population = list(selected)


if __name__ == "__main__":
    # parameters for solutions of the form (h, h)
    parameters_single = {
        "starting_population": [Polytope.cross_polytope(2)],
        "generate_all_mutations": _generate_all_mutations_single,
        "generate_all_crossovers": _generate_all_crossovers_single,
        "remove_unfit": _remove_unfit_single,
        "fitness_function": _calculate_fitness_function_single,
        "p_mutation": 1,
        "max_pop": 30,
        "min_pop": 5,
        "penalty_factor": 1.05,
    }

    solutions = genetic_search(parameters=parameters_single)

    # Uncomment to generate extra solutions from the one found
    # solutions = expand_solutions(set((solutions,)), parameters_single, 10)
    print(solutions)
