# Code modified from the dexom-python project enum_functions modules

# Core imports
import time
from typing import TextIO, Union
from warnings import catch_warnings, filterwarnings, resetwarnings, warn

# External Imports
import cobra
from cobra.exceptions import OptimizationError
from dexom_python.imat_functions import imat, ImatException
from dexom_python.result_functions import write_solution
from dexom_python.model_functions import load_reaction_weights, read_model, check_model_options, \
    check_threshold_tolerance
from dexom_python.enum_functions.enumeration import EnumSolution, create_enum_variables, read_prev_sol
from dexom_python.enum_functions.icut_functions import create_icut_constraint
from dexom_python.enum_functions.maxdist_functions import create_maxdist_constraint, create_maxdist_objective
from dexom_python.default_parameter_values import DEFAULT_VALUES
import numpy as np
import pandas as pd
import six


class EnumIterSolution(object):
    """
    Class for solutions of the enumeration methods iterator

    :param solution: Cobra Solution Object
    :type solution: cobra.Solution
    :param binary: Binary array of reaction activity (0 for inactive, 1 for active)
    :type binary: np.Array
    :param objective_value: Objective value returned by the solver at the end of optimization
    :type objective_value: float
    :param error: Whether an error was produced during the iteration
    :type error: bool
    """

    def __init__(self, solution, binary, objective_value, error=False):
        """Generator Function
        """
        self.solution = solution
        self.binary = binary
        self.objective_value = objective_value
        self.error = error


# region Enumerator Classes
class DiversityEnumIterator:
    """
    Iterator to iterate through the enumerated context specific models created with the Diversity Enum method of dexom

    :param model: Base model from which to create the context specific models
    :type model: cobra.Model
    :param reaction_weights: Weights associated with each reaction,
    :type reaction_weights: pd.Series
    :param prev_sol: A previous imat solution
    :type prev_sol: cobra.Solution
    :param epsilon: Threshold for reaction to be considered active
    :type epsilon: float
    :param threshold: Threshold for reaction to be considered on
    :type threshold: float
    :param obj_tol: Variation allowed in objective values for the solution
    :type obj_tol: float
    :param maxiter: Maximum number of solutions to search for
    :type maxiter: int
    :param dist_anneal: Parameter which influences the probability of selecting reaction
    :type: dist_anneal: float
    :param out_path: Whether to save solutions if save is True
    :type out_path: str
    :param icut: Whether icut constraints should be applied
    :type icut: bool
    :param full: Whether the full-DEXOM implementation should be used
    :type full: bool
    :param save: Whether every solution to the iMAT problem would be saved
    :type save: bool
    :param verbose: Whether a verbose output is desired
    :type verbose: bool
    """

    def __init__(self, model: cobra.Model, reaction_weights: pd.Series, prev_sol: cobra.Solution = None,
                 epsilon: float = DEFAULT_VALUES['epsilon'], threshold: float = DEFAULT_VALUES['threshold'],
                 obj_tol: float = DEFAULT_VALUES['obj_tol'], maxiter: int = DEFAULT_VALUES['maxiter'],
                 dist_anneal: float = DEFAULT_VALUES['dist_anneal'], out_path: TextIO = 'enum_dexom',
                 icut: bool = True, full: bool = False, save: bool = False,
                 verbose=True):
        """Constructor method
        """
        # Set properties on self from arguments
        self.model = model
        self.reaction_weights = reaction_weights
        self.prev_sol = prev_sol
        self.eps = epsilon
        self.thr = threshold
        self.obj_tol = obj_tol
        self.maxiter = maxiter
        self.dist_anneal = dist_anneal
        self.out_path = out_path
        self.icut = icut
        self.full = full
        self.save = save
        self.verbose = verbose
        # Check Threshold Tolerance
        check_threshold_tolerance(model=self.model, epsilon=self.eps, threshold=self.thr)

    def __iter__(self):
        """Iter Method
        """
        # Start the number of iterations at 1 to reflect dexom python code
        self.iter = 1
        if self.prev_sol is None:
            self.prev_sol = imat(self.model, self.reaction_weights, epsilon=self.eps,
                                 threshold=self.thr, full=self.full)
        else:
            self.model = create_enum_variables(model=self.model, reaction_weights=self.reaction_weights, eps=self.eps,
                                               thr=self.thr, full=self.full)
        self.tol = self.model.solver.configuration.tolerances.feasibility
        self.prev_sol_bin = (np.abs(self.prev_sol.fluxes) >= self.thr - self.tol).values.astype(int)
        # Will start the same, but lag 1 behind so if an error is thrown by the optimizer, these represent
        # a safe state to return to
        self.safe_prev_sol = self.prev_sol
        self.safe_prev_sol_bin = self.prev_sol_bin
        # This is preserved for cleaning up the model
        self.icut_constraints = []
        # Preserve the optimality of the original solution
        self.opt_const = create_maxdist_constraint(self.model, self.reaction_weights, self.prev_sol, self.obj_tol,
                                                   'dexom_optimality', full=self.full)
        self.model.solver.add(self.opt_const)
        return self

    def __next__(self):
        """Next method
        """
        # If completed all iterations, stop iteration
        if self.iter > self.maxiter:
            # Complete some cleanup of the model
            self.model.solver.remove([const for const in self.icut_constraints
                                      if const in self.model.solver.constraints])
            self.model.solver.remove(self.opt_const)
            raise StopIteration
        # Record start time for timeout
        t0 = time.perf_counter()
        # set error flag to False
        error = False
        # If icut, add the icut constraint to prevent the algorithm from finding duplicate solutions
        if self.icut:
            const = create_icut_constraint(self.model, reaction_weights=self.reaction_weights, threshold=self.thr,
                                           prev_sol=self.prev_sol, name='icut_' + str(self.iter), full=self.full)
            self.model.solver.add(const)
            self.icut_constraints.append(const)
        # Randomly select reactions with nonzero weight for the distance maximization step
        tempweights = {}
        i = 0
        for rid, weight in six.iteritems(self.reaction_weights):
            if np.random.random() > self.dist_anneal ** self.iter and weight != 0:
                tempweights[rid] = weight
                i += 1
        objective = create_maxdist_objective(self.model, tempweights, self.prev_sol, self.prev_sol_bin, full=self.full)
        self.model.objective = objective
        t2 = time.perf_counter()
        if self.verbose:
            print('Time before optimizing in iteration ' + str(self.iter) + ":", t2 - t0)
        with catch_warnings():
            filterwarnings("error")
            try:
                # with model:
                prev_safe_sol = self.prev_sol

                self.prev_sol = self.model.optimize()
                self.prev_sol_bin = (np.abs(self.prev_sol.fluxes) >= self.thr - self.tol).values.astype(int)
                # Since this iteration worked, save the previous solution as safe
                self.safe_prev_sol = prev_safe_sol
                if self.save:
                    write_solution(self.model, self.prev_sol, self.thr,
                                   filename=self.out_path + '_solution_' + time.strftime('%Y%m%d-%H%M%S') + '.csv')
                    t1 = time.perf_counter()
                    if self.verbose:
                        print('time for optimizing in iteration ' + str(self.iter) + ':', t1 - t2)
            except UserWarning as w:
                resetwarnings()
                error = True
                if 'time_limit' in str(w):
                    print('The solver has reached the timelimit in iteration %i. If this happens frequently, there may '
                          'be too many constraints in the model. Alternatively, you can try modifying solver '
                          'parameters such as the feasibility tolerance or the MIP gap tolerance.' % self.iter)
                    warn('Solver status is "time_limit" in iterations %i' % self.iter)
                elif 'infeasible' in str(w):
                    print('The solver has encountered an infeasible optimization in iteration %i. If this happens '
                          'frequently, there may be a problem with the starting solution. Alternatively, you can try '
                          'modifying solver parameters such as the feasibility tolerance or the MIP gap tolerance.'
                          % self.iter)
                    warn('Solver status is "infeasible" in iteration %i' % self.iter)
                else:
                    print("An unexpected error has occurred during the solver call in iteration %i." % self.iter)
                    warn(w)
            except OptimizationError as e:
                resetwarnings()
                error = True
                self.prev_sol = self.safe_prev_sol
                print("An unexpected error has occurred during the solver call in iteration %i." % self.iter)
                warn(str(e), UserWarning)
        self.iter += 1
        # Create solution object and return it
        solution = EnumIterSolution(self.prev_sol, self.prev_sol_bin, self.prev_sol.objective_value, error)
        stats = {'selected_reactions': i,
                 'time': t1 - t0}
        return solution, stats


class IcutEnumIterator:
    """
    Iterator to iterate through the enumerated context specific models created with the icut Enum method of dexom

    :param model: Base model from which to create the context specific models
    :type model: cobra.Model
    :param reaction_weights: Weights associated with each reaction,
    :type reaction_weights: pd.Series
    :param prev_sol: A previous imat solution
    :type prev_sol: cobra.Solution
    :param epsilon: Threshold for reaction to be considered active
    :type epsilon: float
    :param threshold: Threshold for reaction to be considered on
    :type threshold: float
    :param obj_tol: Variation allowed in objective values for the solution
    :type obj_tol: float
    :param maxiter: Maximum number of solutions to search for
    :type maxiter: int
    :param full: Whether the full-DEXOM implementation should be used
    :type full: bool
    :param verbose: Whether a verbose output is desired
    :type verbose: bool
    """

    def __init__(self, model: cobra.Model, reaction_weights: pd.Series, prev_sol: Union[cobra.Solution, None] = None,
                 epsilon: float = DEFAULT_VALUES['epsilon'], threshold: float = DEFAULT_VALUES['threshold'],
                 obj_tol: float = DEFAULT_VALUES['obj_tol'], maxiter: int = DEFAULT_VALUES['maxiter'],
                 full: bool = False, verbose: bool = False):
        """Constructor method
        """
        self.model = model
        self.reaction_weights = reaction_weights
        self.prev_sol = prev_sol
        self.eps = epsilon
        self.thr = threshold
        self.obj_tol = obj_tol
        self.maxiter = maxiter
        self.full = full
        self.verbose = verbose
        check_threshold_tolerance(model=self.model, epsilon=self.eps, threshold=self.thr)

    def __iter__(self):
        """Iter method
        """
        if self.prev_sol is None:
            self.prev_sol = imat(self.model, self.reaction_weights, epsilon=self.eps, threshold=self.thr,
                                 full=self.full)
        else:
            self.model = create_enum_variables(model=self.model, reaction_weights=self.reaction_weights,
                                               eps=self.eps, thr=self.thr, full=self.full)
        self.tol = self.model.solver.configuration.tolerances.feasibility
        self.prev_sol_binary = (np.abs(self.prev_sol.fluxes) >= self.thr - self.tol).values.astype(int)
        self.objective_value = self.prev_sol.objective_value
        self.optimal_objective_value = self.objective_value - self.obj_tol * self.objective_value
        self.safe_prev_sol = self.prev_sol
        self.icut_constraints = []
        self.iter = 1
        return self

    def __next__(self):
        """Next method
        """
        # If all iterations complete, stop iteration
        if self.iter > self.maxiter:
            # Perform some cleanup
            self.model.solver.remove([const for const in self.icut_constraints if
                                      const in self.model.solver.constraints])
            # Stop the iteration
            raise StopIteration
        t0 = time.perf_counter()
        # Set error flag to false
        error = False
        const = create_icut_constraint(self.model, self.reaction_weights, self.thr, self.prev_sol,
                                       name='icut_' + str(self.iter), full=self.full)
        self.model.solver.add(const)
        with catch_warnings():
            try:
                # Store the current self.prev_sol temporarily, to update the self.safe_prev_sol if the imat works
                prev_safe_sol = self.prev_sol
                self.prev_sol = imat(self.model, self.reaction_weights, epsilon=self.eps, threshold=self.thr,
                                     full=self.full)

                t1 = time.perf_counter()
                if self.verbose:
                    print('time for iteration ' + str(self.iter) + ':', t1 - t0)
                if self.prev_sol.objective_value >= self.optimal_objective_value:
                    self.safe_prev_sol = prev_safe_sol
                    self.prev_sol_binary = (np.abs(self.prev_sol.fluxes) >= self.thr - self.tol).values.astype(int)
                else:
                    # Need to also perform cleanup here
                    self.model.solver.remove([const for const in self.icut_constraints if
                                              const in self.model.solver.constraints])
                    raise StopIteration
            except UserWarning as w:
                resetwarnings()
                error = True
                self.prev_sol = self.safe_prev_sol
                if 'time_limit' in str(w):
                    print('The solver has reached the timelimit in iteration %i. If this happens frequently, there may '
                          'be too many constraints in the model. Alternatively, you can try modifying solver '
                          'parameters such as the feasibility tolerance or the MIP gap tolerance.' % self.iter)
                    warn('Solver status is "time_limit" in iteration %i' % self.iter)
                elif 'infeasible' in str(w):
                    print('The solver has encountered an infeasible optimization in iteration %i. If this happens '
                          'frequently, there may be a problem with the starting solution. Alternatively, you can try '
                          'modifying solver parameters such as the feasibility tolerance or the MIP gap tolerance.'
                          % self.iter)
                    warn('Solver status is "infeasible" in iteration %i' % self.iter)
                else:
                    print('An unexpected error has occurred during the solver call in iteration %i.' % self.iter)
                    warn(w)
            except OptimizationError as e:
                resetwarnings()
                error = True
                self.prev_sol = self.safe_prev_sol
                print('An Unexpected Error has Occurred during the solver call in iteration %i.' % self.iter)
                warn(str(e), UserWarning)

        solution = EnumIterSolution(self.prev_sol, self.prev_sol_binary, self.objective_value, error)
        if self.verbose:
            print("Completed iteration %i" % self.iter)
        stats = {"time": t1 - t0}
        return solution, stats


class MaxDistEnumIterator:
    """
    Iterator to iterate through the enumerated context specific models created with the maxdist Enum method of dexom

    :param model: Base model from which to create the context specific models
    :type model: cobra.Model
    :param reaction_weights: Weights associated with each reaction,
    :type reaction_weights: pd.Series
    :param prev_sol: A previous imat solution
    :type prev_sol: cobra.Solution
    :param epsilon: Threshold for reaction to be considered active
    :type epsilon: float
    :param threshold: Threshold for reaction to be considered on
    :type threshold: float
    :param obj_tol: Variation allowed in objective values for the solution
    :type obj_tol: float
    :param maxiter: Maximum number of solutions to search for
    :type maxiter: int
    :param full: Whether the full-DEXOM implementation should be used
    :type full: bool
    :param icut: Whether to use icut constraints in the enumeration
    :type icut: bool
    :param only_ones: Whether only the ones in the binary solution are used for distance calculation
    :type only_ones: bool
    :param verbose: Whether a verbose output is desired
    :type verbose: bool
    """

    def __init__(self, model: cobra.Model, reaction_weights: pd.Series, prev_sol: cobra.Solution = None,
                 epsilon: float = DEFAULT_VALUES['epsilon'], threshold: float = DEFAULT_VALUES["threshold"],
                 obj_tol: float = DEFAULT_VALUES["obj_tol"], maxiter: int = DEFAULT_VALUES["maxiter"],
                 icut: bool = True, full: bool = False, only_ones: bool = False, verbose=True):
        """Constructor Method
        """
        # Check tolerance
        check_threshold_tolerance(model=model, epsilon=epsilon, threshold=threshold)
        # Store the input values in the object
        self.model = model
        self.reaction_weights = reaction_weights
        self.prev_sol = prev_sol
        self.eps = epsilon
        self.thr = threshold
        self.obj_tol = obj_tol
        self.maxiter = maxiter
        self.icut = icut
        self.full = full
        self.only_ones = only_ones
        self.verbose = verbose

    def __iter__(self):
        """Iter method, setting up iteration
        """
        if self.prev_sol is None:
            self.prev_sol = imat(self.model, self.reaction_weights, epsilon=self.eps, thr=self.thr, full=self.full)
        else:
            self.model = create_enum_variables(model=self.model, reaction_weights=self.reaction_weights,
                                               eps=self.eps, thr=self.thr, full=self.full)
        tol = self.model.solver.configuration.tolerances.feasibility
        self.icut_constraints = []
        self.prev_safe_sol = self.prev_sol
        self.prev_sol_bin = (np.abs(self.prev_sol.fluxes) >= self.thr - self.tol).values.astype(int)
        # Add optimality constraintL the new objective value must be equal to the previous objective values
        opt_const = create_maxdist_constraint(self.model, self.reaction_weights, self.prev_sol, self.obj_tol,
                                              name="maxdist_optimality", full=self.full)
        self.model.solver.add(opt_const)
        self.iter = 1
        return self

    def __next__(self):
        """Next method
        """
        # If maxiter number of iterations completed, stop iteration
        if self.iter > self.maxiter:
            # Complete some cleanup of the model
            self.model.solver.remove([const for const in self.icut_constraints
                                      if const in self.model.solver.constraints])
            self.model.solver.remove(self.opt_const)
            raise StopIteration
        t0 = time.perf_counter()
        error = False
        if self.icut:
            # Adding the icut constraint to prevent the algorithm from finding the same solutions
            const = create_icut_constraint(self.model, self.reaction_weights, self.thr, self.prev_sol,
                                           name='icut_' + str(self.iter), full=self.full)
            self.model.solver.add(const)
            self.icut_constraints.append(const)
        # Defining the objective: minimize the number of overlapping ones and zeroes
        objective = create_maxdist_objective(self.model, self.reaction_weights, self.prev_sol, self.prev_sol_bin,
                                             self.only_ones, self.full)
        self.model.objective = objective
        with catch_warnings():
            filterwarnings('error')
            try:
                prev_safe_sol = self.prev_sol
                self.prev_sol = self.model.optimize()
                self.prev_sol_bin = (np.abs(self.prev_sol.fluxes) >= self.thr - self.tol).values.astype(int)
                # Since this iteration didn't throw an error, save the previous solution as safe
                self.safe_prev_sol = prev_safe_sol
                t1 = time.perf_counter()
                if self.verbose:
                    print('time for iteration ' + str(self.iter) + ':', t1 - t0)
            except UserWarning as w:
                resetwarnings()
                error = True
                self.prev_sol = self.prev_safe_sol
                if 'time_limit' in str(w):
                    print('The solver has reached the timelimit in iteration %i. If this happens frequently, there may '
                          'be too many constraints in the model. Alternatively, you can try modifying solver '
                          'parameters such as the feasibility tolerance or the MIP gap tolerance.' % self.iter)
                    warn('Solver status is "time_limit" in iteration %i' % self.iter)
                elif 'infeasible' in str(w):
                    print('The solver has encountered an infeasible optimization in iteration %i. If this happens '
                          'frequently, there may be a problem with the starting solution. Alternatively, you can try '
                          'modifying solver parameters such as the feasibility tolerance or the MIP gap tolerance.'
                          % self.iter)
                    warn('Solver status is "infeasible" in iteration %i' % self.iter)
                else:
                    print('An unexpected error has occurred during the solver call in iteration %i.' % self.iter)
                    warn(w)
            except OptimizationError as e:
                resetwarnings()
                error = True
                self.prev_sol = self.prev_safe_sol
                print("An unexpected error has occurred during the solver call in iteration %i." % self.iter)
                warn(str(e), UserWarning)
        self.iter += 1
        solution = EnumIterSolution(self.prev_sol, self.prev_sol_bin, self.prev_sol.objective_value, error)
        stats = {'time': t1 - t0}
        return solution, stats

# endregion Enumerator Classes
