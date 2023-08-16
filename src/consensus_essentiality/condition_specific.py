"""
Functions to create condition specific models from flux distributions
"""
# External imports
import cobra
import numpy as np
import six


# region Context Specific Model Functions
def enforce_off(model: cobra.Model, solution: cobra.Solution, thr: float) -> cobra.Model:
    """
    Method to remove reactions whose flux is below tolerance
    :param model: GSMM to modify
    :type model: cobra.Model
    :param solution: Cobra solutions with fluxes to use for determining activity of reactions
    :type solution: cobra.Solution
    :param thr: Tolerance, any reaction with flux less than tolerance is knocked out
    :type thr: float
    :return: Updated GSMM
    :rtype: cobra.Model
    """
    updated_model = model.copy()
    fluxes = solution.fluxes
    rxns_to_rem = list(fluxes[np.abs(fluxes) < thr].index)
    for rxn in rxns_to_rem:
        updated_model.reactions.get_by_id(rxn).knock_out()
    return updated_model


def enforce_inactive(model: cobra.Model, solution: cobra.Solution, epsilon: float, low_expr_rxns: list):
    """
    Enforce low flux through reactions with low expression values, whose flux magnitude in the solution is less than
        epsilon
    :param model: GSMM to update
    :type model: cobra.Model
    :param solution: Solution to the optimization problem being used to update the model
    :type solution: cobra.Solution
    :param epsilon: Threshold for a reaction flux, below which the reaction is considered inactive
    :type epsilon: float
    :param low_expr_rxns: List of reactions which are considered to have low expression values
    :type low_expr_rxns: list
    :return: Updated model with inactive reactions forced to be inactive
    :rtype: cobra.Model
    """
    updated_model = model.copy()
    fluxes = solution.fluxes
    low_flux_rxns = list(fluxes[np.abs(fluxes) < epsilon].index)
    for rxn in low_expr_rxns:
        if rxn in low_flux_rxns:
            rxn_obj = model.reactions.get_by_id(rxn)
            rev = rxn_obj.reversibility
            bounds = _force_inactive(epsilon=epsilon, lb=rxn_obj.lower_bound, ub=rxn_obj.upper_bound, reversible=rev)
            updated_model.reactions.get_by_id(rxn).bounds = bounds
    return updated_model


def enforce_active(model: cobra.Model, solution: cobra.Solution, epsilon: float, high_expr_rxns: list):
    """
    Enforce high flux through reactions with high expression values, whose flux magnitude in the solution is greater
        than epsilon
    :param model: GSMM to update
    :type model: cobra.Model
    :param solution: Solution to the optimization problem being used to update the model
    :type solution: cobra.Solution
    :param epsilon: Threshold for a reaction flux, above which the reaction is considered active
    :type epsilon: float
    :param high_expr_rxns: List of reactions which are considered to have high expression values
    :type high_expr_rxns: list
    :return: Updated model with inactive reactions forced to be inactive
    :rtype: cobra.Model
    """
    updated_model = model.copy()
    fluxes = solution.fluxes
    high_flux_rxns = list(fluxes[np.abs(fluxes) > epsilon].index)
    for rxn in high_expr_rxns:
        if rxn in high_flux_rxns:
            rxn_obj = model.reactions.get_by_id(rxn)
            rev = rxn_obj.reversibility
            forward = fluxes[rxn] > 0
            bounds = _force_active(epsilon=epsilon, lb=rxn_obj.lower_bound, ub=rxn_obj.upper_bound, reversible=rev,
                                   forward=forward)
            updated_model.reactions.get_by_id(rxn).bounds = bounds
    return updated_model


def enforce_both(model: cobra.Model, solution: cobra.Solution, epsilon: float, low_expr_rxns: list,
                 high_expr_rxns: list):
    """
    Enforce low flux through expression with low expression values, whose flux magnitude in the solution is less than
        epsilon, and high flux through reactions with high expression values whose flux magnitude in the solution is
        greater than epsilon
    :param model: GSMM to update
    :type model: cobra.Model
    :param solution: Solution to the optimization problem being used to update the model
    :type solution: cobra.Solution
    :param epsilon: Threshold for a reaction flux, below which the reaction is considered inactive, and above which
        a reaction is considered to be active
    :type epsilon: float
    :param low_expr_rxns: List of reactions which are considered to have low expression values
    :type low_expr_rxns: list
    :param high_expr_rxns: List of reactions which are considered to have high expression values
    :type high_expr_rxns: list
    :return: Updated model with active and inactive reactions forced to be active and inactive respectively
    :rtype: cobra.Model
    """
    updated_model = model.copy()
    fluxes = solution.fluxes
    high_flux_rxns = list(fluxes[np.abs(fluxes) > epsilon].index)
    low_flux_rxns = list(fluxes[np.abs(fluxes) > epsilon].index)
    for rxn in low_expr_rxns:
        if rxn in low_flux_rxns:
            rxn_obj = model.reactions.get_by_id(rxn)
            rev = rxn_obj.reversibility
            bounds = _force_inactive(epsilon=epsilon, lb=rxn_obj.lower_bound, ub=rxn_obj.upper_bound, reversible=rev)
            updated_model.reactions.get_by_id(rxn).bounds = bounds
    for rxn in high_expr_rxns:
        if rxn in high_flux_rxns:
            rxn_obj = model.reactions.get_by_id(rxn)
            rev = rxn_obj.reversibility
            forward = fluxes[rxn] > 0
            bounds = _force_active(epsilon=epsilon, lb=rxn_obj.lower_bound, ub=rxn_obj.upper_bound, reversible=rev,
                                   forward=forward)
            updated_model.reactions.get_by_id(rxn).bounds = bounds
    return updated_model


def enforce_inactive_off(model: cobra.Model, solution: cobra.Solution, epsilon: float, thr: float, low_expr_rxns: list):
    """
    Enforce low flux through reactions with low expression values, whose flux magnitude in the solution is less than
        epsilon
    :param model: GSMM to update
    :type model: cobra.Model
    :param solution: Solution to the optimization problem being used to update the model
    :type solution: cobra.Solution
    :param epsilon: Threshold for a reaction flux, below which the reaction is considered inactive
    :type epsilon: float
    :param thr: Threshold for reaction flux, below which the reaction is considered off
    :type thr: float
    :param low_expr_rxns: List of reactions which are considered to have low expression values
    :type low_expr_rxns: list
    :return: Updated model with inactive reactions forced to be inactive
    :rtype: cobra.Model
    """
    updated_model = model.copy()
    fluxes = solution.fluxes
    low_flux_rxns = list(fluxes[np.abs(fluxes) < epsilon].index)
    for rxn in low_expr_rxns:
        if rxn in low_flux_rxns:
            rxn_obj = model.reactions.get_by_id(rxn)
            rev = rxn_obj.reversibility
            bounds = _force_inactive(epsilon=epsilon, lb=rxn_obj.lower_bound, ub=rxn_obj.upper_bound, reversible=rev)
            updated_model.reactions.get_by_id(rxn).bounds = bounds
    rxns_to_rem = list(fluxes[np.abs(fluxes) < thr].index)
    for rxn in rxns_to_rem:
        updated_model.reactions.get_by_id(rxn).knock_out()
    return updated_model


# endregion

# region Context Managers
class EnforceOff:
    """This is a context manager to create a model where all the reactions whose flux is below tolerance are turned off.

    :param model: A genome scale metabolic model to create the context specific model from
    :type model: cobra.Model
    :param solution: Solution returned from the imat methods, with fluxes property
    :type solution: cobra.Solution
    :param thr: Cutoff, below which reactions are considered off and knocked out
    :type thr: float
    """

    def __init__(self, model, solution, thr):
        """
        Constructor method

        :param model: A genome scale metabolic model to create the context specific model from
        :type model: cobra.Model
        :param solution: Solution returned from the imat methods, with fluxes property
        :type solution: cobra.Solution
        :param thr: Cutoff, below which reactions are considered off and knocked out
        :type thr: float
        """
        self.model = model
        self.solution = solution
        self.tol = thr
        self.rxn_bounds = {}

    def __enter__(self):
        """
        Method called on entering context, updates the model and returns it

        :return: Updated model object
        :rtype: cobra.Model
        """
        fluxes = self.solution.fluxes
        rxns_to_rem = list(fluxes[np.abs(fluxes) < self.tol].index)
        for rxn in rxns_to_rem:
            if rxn in self.model.reactions:
                rxn_obj = self.model.reactions.get_by_id(rxn)
                # Save original bounds
                self.rxn_bounds[rxn] = (rxn_obj.lower_bound, rxn_obj.upper_bound)
                # set the upper and lower bound of the reaction to 0 using the knock_out method
                rxn_obj.knock_out()
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Method called upon exiting context, restores models original reaction bounds

        :param exc_type: Type of exception that caused exit
        :param exc_val: Value of exception that caused exit
        :param exc_tb: Traceback of exception that caused exit
        :return: Boolean flag indicating if exceptions were handled
        :rtype: bool
        """
        # Cleanup model
        for rxn, bounds in six.iteritems(self.rxn_bounds):
            self.model.reactions.get_by_id(rxn).bounds = bounds


class EnforceInactive:
    """This is a context manager to create a model where all the low expression reactions whose flux is below epsilon
    have their maximum flux set to epsilon.

    :param model: A genome scale metabolic model to create the context specific model from
    :type model: cobra.Model
    :param solution: Solution returned from the imat methods, with fluxes property
    :type solution: cobra.Solution
    :param epsilon: Cutoff, below which reactions are considered inactive, above which reactions are considered active
    :type epsilon: float
    :param low_expr_rxns: List of reactions which are considered to have low expression values
    :type low_expr_rxns: list
    """

    def __init__(self, model: cobra.Model, solution: cobra.Solution, epsilon: float, low_expr_rxns: list):
        """
        Constructor method

        :param model: A genome scale metabolic model to create the context specific model from
        :type model: cobra.Model
        :param solution: Solution returned from the imat methods, with fluxes property
        :type solution: cobra.Solution
        :param epsilon: Cutoff, below which reactions are considered inactive, above which reactions are considered active
        :type epsilon: float
        :param low_expr_rxns: List of reactions which are considered to have low expression values
        :type low_expr_rxns: list
        """
        self.model = model
        self.solution = solution
        self.epsilon = epsilon
        self.low_expr_rxns = low_expr_rxns
        self.rxn_bounds = {}

    def __enter__(self):
        """
        Method called on entering context, updates the model and returns it

        :return: Updated model object
        :rtype: cobra.Model
        """
        fluxes = self.solution.fluxes
        low_flux_rxns = list(fluxes[np.abs(fluxes) < self.epsilon].index)
        for rxn in self.low_expr_rxns:
            if (rxn in low_flux_rxns) and (rxn in self.model.reactions):
                rxn_obj = self.model.reactions.get_by_id(rxn)
                rev = rxn_obj.reversibility
                # Save original bounds
                self.rxn_bounds[rxn] = (rxn_obj.lower_bound, rxn_obj.upper_bound)
                # Find the new bounds
                bounds = _force_inactive(epsilon=self.epsilon, lb=rxn_obj.lower_bound,
                                         ub=rxn_obj.upper_bound, reversible=rev)
                # set the upper and lower bound of the reaction to bounds returned from the _force_inactive function
                rxn_obj.bounds = bounds
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Method called upon exiting context, restores models original reaction bounds

        :param exc_type: Type of exception that caused exit
        :param exc_val: Value of exception that caused exit
        :param exc_tb: Traceback of exception that caused exit
        :return: Boolean flag indicating if exceptions were handled
        :rtype: bool
        """
        # Cleanup model
        for rxn, bounds in six.iteritems(self.rxn_bounds):
            self.model.reactions.get_by_id(rxn).bounds = bounds


class EnforceActive:
    """This is a context manager to create a model where all the high expression reactions whose flux is above epsilon
    have their minimum flux set to epsilon.

    :param model: A genome scale metabolic model to create the context specific model from
    :type model: cobra.Model
    :param solution: Solution returned from the imat methods, with fluxes property
    :type solution: cobra.Solution
    :param epsilon: Cutoff, below which reactions are considered inactive, above which reactions are considered active
    :type epsilon: float
    :param high_expr_rxns: List of reactions which are considered to have high expression values
    :type high_expr_rxns: list
    """

    def __init__(self, model: cobra.Model, solution: cobra.Solution, epsilon: float, high_expr_rxns: list):
        """
        Constructor method

        :param model: A genome scale metabolic model to create the context specific model from
        :type model: cobra.Model
        :param solution: Solution returned from the imat methods, with fluxes property
        :type solution: cobra.Solution
        :param epsilon: Cutoff, below which reactions are considered inactive, above which reactions are considered active
        :type epsilon: float
        :param high_expr_rxns: List of reactions which are considered to have high expression values
        :type high_expr_rxns: list
        """
        self.model = model
        self.solution = solution
        self.epsilon = epsilon
        self.high_expr_rxns = high_expr_rxns
        self.rxn_bounds = {}

    def __enter__(self):
        """
        Method called on entering context, updates the model and returns it

        :return: Updated model object
        :rtype: cobra.Model
        """
        fluxes = self.solution.fluxes
        high_flux_rxns = list(fluxes[np.abs(fluxes) > self.epsilon].index)
        for rxn in self.high_expr_rxns:
            if (rxn in high_flux_rxns) and (rxn in self.model.reactions):
                rxn_obj = self.model.reactions.get_by_id(rxn)
                rev = rxn_obj.reversibility
                forward = fluxes[rxn] > 0
                # Save original bounds
                self.rxn_bounds[rxn] = (rxn_obj.lower_bound, rxn_obj.upper_bound)
                # Find the new bounds
                bounds = _force_active(epsilon=self.epsilon, lb=rxn_obj.lower_bound,
                                       ub=rxn_obj.upper_bound, reversible=rev,
                                       forward=forward)
                # set the upper and lower bound of the reaction using the bounds from _force_active
                rxn_obj.bounds = bounds
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Method called upon exiting context, restores models original reaction bounds

        :param exc_type: Type of exception that caused exit
        :param exc_val: Value of exception that caused exit
        :param exc_tb: Traceback of exception that caused exit
        :return: Boolean flag indicating if exceptions were handled
        :rtype: bool
        """
        # Cleanup model
        for rxn, bounds in six.iteritems(self.rxn_bounds):
            self.model.reactions.get_by_id(rxn).bounds = bounds


class EnforceBoth:
    """This is a context manager to create a model where all the high expression reactions whose flux is above epsilon
    have their minimum flux set to epsilon, and all the low expression reactions whose flux is below epsilon have their
    maximum flux set to epsilon.

    :param model: A genome scale metabolic model to create the context specific model from
    :type model: cobra.Model
    :param solution: Solution returned from the imat methods, with fluxes property
    :type solution: cobra.Solution
    :param epsilon: Cutoff, below which reactions are considered inactive, above which reactions are considered active
    :type epsilon: float
    :param high_expr_rxns: List of reactions which are considered to have high expression values
    :type high_expr_rxns: list
    :param low_expr_rxns: List of reactions which are considered to have low expression values
    :type low_expr_rxns: list
    """

    def __init__(self, model: cobra.Model, solution: cobra.Solution, epsilon: float, high_expr_rxns: list,
                 low_expr_rxns: list):
        """
        Constructor method

        :param model: A genome scale metabolic model to create the context specific model from
        :type model: cobra.Model
        :param solution: Solution returned from the imat methods, with fluxes property
        :type solution: cobra.Solution
        :param epsilon: Cutoff, below which reactions are considered inactive, above which reactions are considered active
        :type epsilon: float
        :param high_expr_rxns: List of reactions which are considered to have high expression values
        :type high_expr_rxns: list
        :param low_expr_rxns: List of reactions which are considered to have low expression values
        :type low_expr_rxns: list
        """
        self.model = model
        self.solution = solution
        self.epsilon = epsilon
        self.high_expr_rxns = high_expr_rxns
        self.low_expr_rxns = low_expr_rxns
        self.rxn_bounds = {}

    def __enter__(self):
        """
        Method called on entering context, updates the model and returns it

        :return: Updated model object
        :rtype: cobra.Model
        """
        fluxes = self.solution.fluxes
        high_flux_rxns = list(fluxes[np.abs(fluxes) > self.epsilon].index)
        low_flux_rxns = list(fluxes[np.abs(fluxes) < self.epsilon].index)
        for rxn in self.high_expr_rxns:
            if (rxn in high_flux_rxns) and (rxn in self.model.reactions):
                rxn_obj = self.model.reactions.get_by_id(rxn)
                rev = rxn_obj.reversibility
                forward = fluxes[rxn] > 0
                # Save original bounds
                self.rxn_bounds[rxn] = (rxn_obj.lower_bound, rxn_obj.upper_bound)
                # Find the new bounds
                bounds = _force_active(epsilon=self.epsilon, lb=rxn_obj.lower_bound,
                                       ub=rxn_obj.upper_bound, reversible=rev,
                                       forward=forward)
                # set the upper and lower bound of the reaction using the bounds from _force_active
                rxn_obj.bounds = bounds
        for rxn in self.low_expr_rxns:
            if (rxn in low_flux_rxns) and (rxn in self.model.reactions):
                rxn_obj = self.model.reactions.get_by_id(rxn)
                rev = rxn_obj.reversibility
                # Save original bounds
                self.rxn_bounds[rxn] = (rxn_obj.lower_bound, rxn_obj.upper_bound)
                # Find the new bounds
                bounds = _force_inactive(epsilon=self.epsilon, lb=rxn_obj.lower_bound,
                                         ub=rxn_obj.upper_bound, reversible=rev)
                # set the upper and lower bound of the reaction to bounds returned from the _force_inactive function
                rxn_obj.bounds = bounds
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Method called upon exiting context, restores models original reaction bounds

        :param exc_type: Type of exception that caused exit
        :param exc_val: Value of exception that caused exit
        :param exc_tb: Traceback of exception that caused exit
        :return: Boolean flag indicating if exceptions were handled
        :rtype: bool
        """
        # Cleanup model
        for rxn, bounds in six.iteritems(self.rxn_bounds):
            self.model.reactions.get_by_id(rxn).bounds = bounds


class EnforceInactiveOff:
    """This is a context manager to create a model where all the low expression reactions whose flux is below epsilon
    have their maximum flux set to epsilon, and all reactions with flux below tolerance are turned off.

    :param model: A genome scale metabolic model to create the context specific model from
    :type model: cobra.Model
    :param solution: Solution returned from the imat methods, with fluxes property
    :type solution: cobra.Solution
    :param epsilon: Cutoff, below which reactions are considered inactive, above which reactions are considered active
    :type epsilon: float
    :param thr: Cutoff, below which reactions are considered off
    :type thr: float
    :param low_expr_rxns: List of reactions which are considered to have low expression values
    :type low_expr_rxns: list
    """

    def __init__(self, model: cobra.Model, solution: cobra.Solution, epsilon: float, thr: float, low_expr_rxns: list):
        """
        Constructor method

        :param model: A genome scale metabolic model to create the context specific model from
        :type model: cobra.Model
        :param solution: Solution returned from the imat methods, with fluxes property
        :type solution: cobra.Solution
        :param epsilon: Cutoff, below which reactions are considered inactive, above which reactions are considered active
        :type epsilon: float
        :param thr: Cutoff, below which reactions are considered off
        :type thr: float
        :param low_expr_rxns: List of reactions which are considered to have low expression values
        :type low_expr_rxns: list
        """
        self.model = model
        self.solution = solution
        self.epsilon = epsilon
        self.tol = thr
        self.low_expr_rxns = low_expr_rxns
        self.rxn_bounds = {}

    def __enter__(self):
        """
        Method called on entering context, updates the model and returns it

        :return: Updated model object
        :rtype: cobra.Model
        """
        fluxes = self.solution.fluxes
        low_flux_rxns = list(fluxes[np.abs(fluxes) < self.epsilon].index)
        for rxn in self.low_expr_rxns:
            if (rxn in low_flux_rxns) and (rxn in self.model.reactions):
                rxn_obj = self.model.reactions.get_by_id(rxn)
                rev = rxn_obj.reversibility
                # Save original bounds
                self.rxn_bounds[rxn] = (rxn_obj.lower_bound, rxn_obj.upper_bound)
                # Find the new bounds
                bounds = _force_inactive(epsilon=self.epsilon, lb=rxn_obj.lower_bound,
                                         ub=rxn_obj.upper_bound, reversible=rev)
                # set the upper and lower bound of the reaction to bounds returned from the _force_inactive function
                rxn_obj.bounds = bounds
        rxns_to_rem = list(fluxes[np.abs(fluxes) < self.tol].index)
        for rxn in rxns_to_rem:
            if rxn in self.model.reactions:
                rxn_obj = self.model.reactions.get_by_id(rxn)
                # Save original bounds (if not already saved)
                if rxn not in self.rxn_bounds:
                    self.rxn_bounds[rxn] = (rxn_obj.lower_bound, rxn_obj.upper_bound)
                # set the upper and lower bound of the reaction to 0 using the knock_out method
                rxn_obj.knock_out()
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Method called upon exiting context, restores models original reaction bounds

        :param exc_type: Type of exception that caused exit
        :param exc_val: Value of exception that caused exit
        :param exc_tb: Traceback of exception that caused exit
        :return: Boolean flag indicating if exceptions were handled
        :rtype: bool
        """
        # Cleanup model
        for rxn, bounds in six.iteritems(self.rxn_bounds):
            self.model.reactions.get_by_id(rxn).bounds = bounds


# endregion

# region Helper Functions
# There is some redundancy to this function
def _force_inactive(epsilon: float, lb: float, ub: float, reversible: bool) -> tuple[float, float]:
    """
    Function to force reaction to have flux less than epsilon
    :param epsilon: Cutoff value for reaction to be considered active
    :type epsilon: float
    :param lb: Lower bound of the reaction flux
    :type lb: float
    :param ub: Upper bound of the reaction flux
    :type ub: float
    :param reversible: Whether the reaction is reversible or not
    :type reversible: bool
    :return: Updated bounds
    :rtype: tuple[float,float]
    """
    if epsilon < 0:
        raise ValueError("Epsilon must not be negative")
    updated_lb = lb
    updated_ub = ub
    # If the reaction is reversible, its bounds range spans 0
    if reversible:
        # If the epsilon is more restrictive than the old lb, apply it
        if (-epsilon) >= lb:
            updated_lb = -epsilon
        # Otherwise, just keep the former lower bound
        else:
            pass
        # If the epsilon is more restrictive than the old ub, apply it
        if epsilon <= ub:
            updated_ub = epsilon
        # Otherwise, just keep the former upper bound
        else:
            pass
    else:
        # This is the non-reversible case
        if ub > 0:
            # If the upper bound is positive (the rxn is forward), and the epsilon is more restrictive
            #       but will not violate the lb<=ub if used as ub, set it as the upper bound
            if lb <= epsilon <= ub:
                updated_ub = epsilon
            elif epsilon < ub and epsilon < lb:
                raise ValueError("The lower bound is positive and greater than epsilon")
            else:
                # This case means the upperbound is already more restrictive than epsilon
                pass
        if lb < 0:
            # If the lower-bound is negative (the reaction is reversed), and the epsilon is more restrictive,
            #       but will not violate the lb <= ub if used as lb, set it as the lower bound
            if lb <= -epsilon <= ub:
                updated_lb = -epsilon
            elif -epsilon > lb and -epsilon > ub:
                raise ValueError("The upperbound is negative and greater than negative epsilon")
            else:
                # This case means the old lb is already more restrictive than the epsilon
                pass
    return updated_lb, updated_ub


# This function is for forcing reactions to be above epsilon
def _force_active(epsilon: float, lb: float, ub: float, reversible: bool, forward: bool) -> tuple[float, float]:
    """
    Function to force reaction to have flux greater than epsilon
    :param epsilon: Cutoff value for reaction to be considered active
    :type epsilon: float
    :param lb: Lower bound of the reaction flux
    :type lb: float
    :param ub: Upper bound of the reaction flux
    :type ub: float
    :param reversible: Whether the reaction is reversible or not
    :type reversible: bool
    :param forward: Whether the reaction is forward or reverse in the flux solution (forward if positive flux)
    :type forward: bool
    :return: Updated bounds
    :rtype: tuple[float,float]
    """
    updated_lb = lb
    updated_ub = ub
    if epsilon < 0:
        raise ValueError("Epsilon must not be negative")
    if reversible:
        if forward:
            if epsilon <= ub:
                updated_lb = epsilon
            else:
                raise ValueError("The reaction is reversible and positive epsilon is greater than the upper bound")
        else:
            if -epsilon >= lb:
                updated_ub = -epsilon
            else:
                raise ValueError("The reaction is reversible and negative epsilon is less than the lower bounds")
    else:
        if ub > 0:
            # This is the forward reaction case
            if lb <= epsilon <= ub:
                updated_lb = epsilon
            elif epsilon > ub:
                raise ValueError("The reaction is forward, but epsilon is greater than the upper bound")
            else:
                # This is the case where the lower bound is already greater than epsilon, and it won't violate lb<=ub
                pass
        if lb < 0:
            # This is the reverse reaction case
            if lb <= -epsilon <= ub:
                updated_ub = -epsilon
            elif -epsilon < lb:
                raise ValueError("The reaction is reverse, but epsilon is less than the lower bound")
            else:
                # This is the case where the upper bound is already less than negative epsilon, and won't violate lb<=ub
                pass
    return updated_lb, updated_ub
# endregion
